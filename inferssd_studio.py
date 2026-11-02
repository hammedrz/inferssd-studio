#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""InferSSD Studio: load a mixture-of-experts model whose experts stay on the SSD, chat with it, serve it.

A GTK window around one inferssd server (engine.py). The Chat tab is the chat app built into that
server (llama.cpp's web UI); the Serve tab sets up and shows its OpenAI-compatible API.
An unofficial app, not affiliated with WiCi AI, the authors of InferSSD.
"""

import json
import os
import secrets
import signal
import sys
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
gi.require_version("WebKit", "6.0")
from gi.repository import Adw, Gdk, Gio, GLib, GLibUnix, Gtk, WebKit  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engine import DATA_DIR, Engine, Settings, find_inferssd, find_models, inferssd_at, lan_address  # noqa: E402

APP_ID = "io.github.hammedrz.InferSSDStudio"
VERSION = "0.1.0"
WEBSITE = "https://github.com/hammedrz/inferssd-studio"
LOG_LIMIT = 300_000  # characters of server log kept in the window
CPUS = os.cpu_count() or 8

CSS = """
.log-view { font-family: monospace; font-size: 9pt; padding: 8px; }
.example-view { font-family: monospace; font-size: 9.5pt; padding: 10px; border-radius: 8px; }
.status-ready { color: @success_color; }
.status-busy { color: @accent_color; }
.status-failed { color: @error_color; }
"""


def copy_button(get_text, window):
    button = Gtk.Button(icon_name="edit-copy-symbolic", valign=Gtk.Align.CENTER, tooltip_text="Copy")
    button.add_css_class("flat")

    def copy(_):
        window.get_clipboard().set(get_text())
        window.toast("Copied")

    button.connect("clicked", copy)
    return button


class Window(Adw.ApplicationWindow):
    def __init__(self, app):
        super().__init__(application=app, title="InferSSD Studio", default_width=1180, default_height=820)
        self.settings = Settings.load()
        self.engine = Engine(lambda s, d: GLib.idle_add(self.on_engine_state, s, d), self.queue_log)
        self.models = []
        self.updating_models = False
        self.web = None
        self.closing = False
        self.pending_log = []
        self.log_lock = threading.Lock()

        self.toasts = Adw.ToastOverlay()
        toolbar = Adw.ToolbarView()
        self.toasts.set_child(toolbar)
        self.set_content(self.toasts)

        self.stack = Adw.ViewStack()
        header = Adw.HeaderBar()
        header.set_title_widget(Adw.ViewSwitcher(stack=self.stack, policy=Adw.ViewSwitcherPolicy.WIDE))
        toolbar.add_top_bar(header)
        toolbar.set_content(self.stack)

        self.model_names = Gtk.StringList()
        self.picker = Gtk.DropDown(model=self.model_names, tooltip_text="Model to load")
        self.picker.set_size_request(330, -1)
        self.picker.connect("notify::selected", self.on_model_picked)
        self.load_button = Gtk.Button(label="Load")
        self.load_button.add_css_class("suggested-action")
        self.load_button.connect("clicked", self.on_load_clicked)
        header.pack_start(self.picker)
        header.pack_start(self.load_button)

        menu = Gio.Menu()
        menu.append("Refresh models", "win.refresh")
        menu.append("Add model folder…", "win.add-folder")
        menu.append("Open server run folder", "win.open-run")
        menu.append("About InferSSD Studio", "win.about")
        header.pack_end(Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu, tooltip_text="Menu"))
        self.status = Gtk.Label(label="No model loaded")
        header.pack_end(self.status)
        for name, handler in (("refresh", self.refresh_models), ("add-folder", self.on_add_folder),
                              ("open-run", self.on_open_run), ("about", self.on_about),
                              ("show-chat", lambda: self.stack.set_visible_child_name("chat")),
                              ("show-serve", lambda: self.stack.set_visible_child_name("serve"))):
            action = Gio.SimpleAction.new(name, None)
            action.connect("activate", lambda *_, h=handler: h())
            self.add_action(action)
        # Tab shortcuts in the capture phase: the chat page would otherwise take the keys while it has focus.
        tabs = Gtk.ShortcutController(propagation_phase=Gtk.PropagationPhase.CAPTURE)
        for keys, action in (("<Control>1", "win.show-chat"), ("<Control>2", "win.show-serve")):
            tabs.add_shortcut(Gtk.Shortcut(trigger=Gtk.ShortcutTrigger.parse_string(keys),
                                           action=Gtk.NamedAction.new(action)))
        self.add_controller(tabs)

        self.stack.add_titled_with_icon(self.build_chat_page(), "chat", "Chat", "chat-message-new-symbolic")
        self.stack.add_titled_with_icon(self.build_serve_page(), "serve", "Serve", "network-server-symbolic")

        self.connect("close-request", self.on_close_request)
        GLib.timeout_add(200, self.flush_log)
        self.refresh_models()
        self.update_api_rows()
        if not self.update_engine_row():
            self.toast("inferssd was not found: install it, or choose its folder on the Serve tab", timeout=8)

    # ---------------------------------------------------------------- pages

    def build_chat_page(self):
        self.chat_stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        load = Gtk.Button(label="Load Model", halign=Gtk.Align.CENTER)
        load.add_css_class("pill")
        load.add_css_class("suggested-action")
        load.connect("clicked", self.on_load_clicked)
        empty = Adw.StatusPage(icon_name="drive-harddisk-solidstate-symbolic", title="No model loaded",
                               description="Pick a model in the header bar and load it. Its experts stay on "
                                           "the SSD; inferssd reads the ones each token needs.", child=load)
        self.chat_stack.add_named(empty, "empty")
        self.loading_page = Adw.StatusPage(title="Loading…", child=Adw.Spinner(width_request=48, height_request=48),
                                           description="Indexing the experts and loading the dense weights.")
        self.chat_stack.add_named(self.loading_page, "loading")
        show_log = Gtk.Button(label="Show Log", halign=Gtk.Align.CENTER)
        show_log.add_css_class("pill")
        show_log.connect("clicked", lambda _: self.stack.set_visible_child_name("serve"))
        self.failed_page = Adw.StatusPage(icon_name="dialog-error-symbolic", title="The model did not load",
                                          child=show_log)
        self.chat_stack.add_named(self.failed_page, "failed")
        return self.chat_stack

    def build_serve_page(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.banner = Adw.Banner(title="Settings changed. Reload the model to apply them.", button_label="Reload")
        self.banner.connect("button-clicked", lambda _: self.reload())
        box.append(self.banner)
        page = Adw.PreferencesPage(vexpand=True)
        box.append(page)

        api = Adw.PreferencesGroup(title="OpenAI-Compatible API",
                                   description="Any OpenAI client can use the loaded model at these addresses.")
        self.state_row = Adw.ActionRow(title="Status")
        self.url_row = Adw.ActionRow(title="Base URL", subtitle_selectable=True)
        self.url_row.add_suffix(copy_button(lambda: self.base_url(), self))
        self.model_row = Adw.ActionRow(title="Model ID", subtitle_selectable=True)
        self.model_row.add_suffix(copy_button(lambda: self.model_row.get_subtitle(), self))
        self.lan_row = Adw.ActionRow(title="Local network URL", subtitle_selectable=True)
        self.lan_row.add_suffix(copy_button(lambda: self.lan_row.get_subtitle(), self))
        for row in (self.state_row, self.url_row, self.model_row, self.lan_row):
            api.add(row)
        page.add(api)

        engine = Adw.PreferencesGroup(title="inferssd",
                                      description="The inferssd that runs the models: its installer's, or a source "
                                                  "checkout set up with .venv and engine-resources.json. DeepSeek-V2 "
                                                  "and Gemma 4 need the inferssd-gemma4 fork.")
        self.engine_row = Adw.ActionRow(title="Location", subtitle_selectable=True)
        automatic = Gtk.Button(icon_name="edit-undo-symbolic", valign=Gtk.Align.CENTER,
                               tooltip_text="Find it automatically")
        automatic.add_css_class("flat")
        automatic.connect("clicked", lambda _: self.set_inferssd(""))
        choose = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose.connect("clicked", lambda _: self.on_choose_inferssd())
        self.engine_row.add_suffix(automatic)
        self.engine_row.add_suffix(choose)
        engine.add(self.engine_row)
        page.add(engine)

        s = self.settings
        server = Adw.PreferencesGroup(title="Server Settings",
                                      description="These apply when the model is loaded or reloaded.")
        self.port_row = Adw.SpinRow.new_with_range(1024, 65535, 1)
        self.port_row.set_title("Port")
        self.port_row.set_value(s.port)
        self.context_row = Adw.SpinRow.new_with_range(512, 262144, 512)
        self.context_row.set_title("Context length")
        self.context_row.set_subtitle("Tokens of conversation the server keeps; more uses more RAM")
        self.context_row.set_value(s.context)
        self.limit_row = Adw.SwitchRow(title="Limit memory", active=s.memory_limit,
                                       subtitle="Without a limit the expert cache uses the RAM that is free")
        self.memory_row = Adw.SpinRow.new_with_range(2, 256, 1)
        self.memory_row.set_title("Memory limit (GiB)")
        self.memory_row.set_value(s.memory_gib)
        self.threads_row = Adw.SpinRow.new_with_range(0, CPUS, 1)
        self.threads_row.set_title("CPU threads")
        self.threads_row.set_subtitle("0 lets inferssd choose for this machine")
        self.threads_row.set_value(s.threads)
        self.cache_row = Adw.SwitchRow(title="Disk prompt cache", active=s.prompt_cache,
                                       subtitle="Keeps prompt state across restarts; writes up to 16 GiB "
                                                "a model to the system drive")
        self.key_row = Adw.PasswordEntryRow(title="API key (optional)", text=s.api_key)
        generate = Gtk.Button(icon_name="view-refresh-symbolic", valign=Gtk.Align.CENTER,
                              tooltip_text="Generate a key")
        generate.add_css_class("flat")
        generate.connect("clicked", lambda _: self.key_row.set_text("sk-" + secrets.token_urlsafe(24)))
        self.key_row.add_suffix(generate)
        self.key_row.add_suffix(copy_button(lambda: self.key_row.get_text(), self))
        for row in (self.port_row, self.context_row, self.limit_row, self.memory_row, self.threads_row,
                    self.cache_row, self.key_row):
            server.add(row)
        page.add(server)

        network = Adw.PreferencesGroup(title="Local Network",
                                       description="Other devices on your network reach the API through this "
                                                   "port. Set an API key before turning it on.")
        self.lan_switch = Adw.SwitchRow(title="Serve on local network", active=s.lan)
        self.lan_port_row = Adw.SpinRow.new_with_range(1024, 65535, 1)
        self.lan_port_row.set_title("Network port")
        self.lan_port_row.set_value(s.lan_port)
        network.add(self.lan_switch)
        network.add(self.lan_port_row)
        page.add(network)

        example = Adw.PreferencesGroup(title="Example Request")
        copy_example = copy_button(lambda: self.example_buffer.get_text(*self.example_buffer.get_bounds(), False),
                                   self)
        example.set_header_suffix(copy_example)
        self.example_buffer = Gtk.TextBuffer()
        view = Gtk.TextView(buffer=self.example_buffer, editable=False, monospace=True,
                            wrap_mode=Gtk.WrapMode.CHAR)
        view.add_css_class("example-view")
        view.add_css_class("card")
        example.add(view)
        page.add(example)

        logs = Adw.PreferencesGroup(title="Server Log")
        clear = Gtk.Button(icon_name="edit-clear-all-symbolic", tooltip_text="Clear", valign=Gtk.Align.CENTER)
        clear.add_css_class("flat")
        clear.connect("clicked", lambda _: self.log_buffer.set_text(""))
        logs.set_header_suffix(clear)
        self.log_buffer = Gtk.TextBuffer()
        self.log_end = self.log_buffer.create_mark("end", self.log_buffer.get_end_iter(), False)
        self.log_view = Gtk.TextView(buffer=self.log_buffer, editable=False, monospace=True,
                                     wrap_mode=Gtk.WrapMode.CHAR, cursor_visible=False)
        self.log_view.add_css_class("log-view")
        self.log_scroll = Gtk.ScrolledWindow(child=self.log_view, min_content_height=320)
        self.log_scroll.add_css_class("card")
        logs.add(self.log_scroll)
        page.add(logs)

        for row in (self.port_row, self.context_row, self.memory_row, self.threads_row, self.lan_port_row):
            row.connect("notify::value", self.on_setting_changed)
        for row in (self.limit_row, self.cache_row, self.lan_switch):
            row.connect("notify::active", self.on_setting_changed)
        self.key_row.connect("changed", self.on_setting_changed)
        self.memory_row.set_sensitive(s.memory_limit)
        return box

    # ---------------------------------------------------------------- models

    def refresh_models(self):
        def scan():
            models = find_models(self.settings.model_dirs)
            GLib.idle_add(self.show_models, models)
        threading.Thread(target=scan, daemon=True).start()

    def show_models(self, models):
        self.models = [m for m in models if m.supported]
        skipped = [m for m in models if not m.supported]
        names = [f"{m.name}  ·  {m.size / 1e9:.1f} GB" for m in self.models] or ["No supported models found"]
        chosen = next((i for i, m in enumerate(self.models) if m.path == self.settings.model), 0)
        self.updating_models = True  # the list change selects its first item; keep the saved choice
        self.model_names.splice(0, self.model_names.get_n_items(), names)
        self.picker.set_selected(chosen)
        self.updating_models = False
        self.update_controls()
        self.update_api_rows()
        if skipped:
            self.toast(f"Hidden: {', '.join(m.name for m in skipped)} (no routed experts; inferssd runs "
                       "mixture-of-experts models only)", timeout=6)

    def selected_model(self):
        index = self.picker.get_selected()
        return self.models[index] if 0 <= index < len(self.models) else None

    def on_model_picked(self, *_):
        model = self.selected_model()
        if model and not self.updating_models and model.path != self.settings.model:
            self.settings.model = model.path
            self.settings.save()
            self.update_banner()
            self.update_api_rows()

    def on_add_folder(self):
        dialog = Gtk.FileDialog(title="Add a folder with GGUF models")

        def chosen(dialog, result):
            try:
                folder = dialog.select_folder_finish(result)
            except GLib.Error:
                return
            path = folder.get_path()
            if path and path not in self.settings.model_dirs:
                self.settings.model_dirs.append(path)
                self.settings.save()
                self.refresh_models()

        dialog.select_folder(self, None, chosen)

    # ---------------------------------------------------------------- inferssd

    def update_engine_row(self):
        found = find_inferssd(self.settings.inferssd_dir)
        if found:
            home = str(Path.home())
            root = str(found.root)
            self.engine_row.set_subtitle("~" + root[len(home):] if root.startswith(home + "/") else root)
        else:
            self.engine_row.set_subtitle("Not found: install inferssd, or choose its folder")
        return found

    def set_inferssd(self, folder):
        self.settings.inferssd_dir = folder
        self.settings.save()
        if not self.update_engine_row() and not folder:
            self.toast("No inferssd found automatically: choose its folder")

    def on_choose_inferssd(self):
        dialog = Gtk.FileDialog(title="Choose the inferssd folder")

        def chosen(dialog, result):
            try:
                folder = dialog.select_folder_finish(result)
            except GLib.Error:
                return
            path = folder.get_path()
            if path and inferssd_at(path):
                self.set_inferssd(path)
            else:
                self.toast("No inferssd in that folder: it needs venv/bin/inferssd or .venv/bin/inferssd, "
                           "and engine-resources.json", timeout=6)

        dialog.select_folder(self, None, chosen)

    # ---------------------------------------------------------------- engine

    def on_load_clicked(self, *_):
        state = self.engine.state
        if state in ("loading", "ready"):
            threading.Thread(target=self.engine.stop, daemon=True).start()
            return
        model = self.selected_model()
        if not model:
            self.toast("No supported model to load: add a folder from the menu")
            return
        self.settings.model = model.path
        self.settings.save()
        try:
            self.engine.start(model, self.settings)
        except (RuntimeError, OSError) as error:
            self.toast(str(error), timeout=6)

    def reload(self):
        def run():
            self.engine.stop()
            GLib.idle_add(self.on_load_clicked)
        threading.Thread(target=run, daemon=True).start()

    def on_engine_state(self, state, detail):
        self.status.remove_css_class("status-ready")
        self.status.remove_css_class("status-busy")
        self.status.remove_css_class("status-failed")
        if state == "loading":
            self.status.set_label(f"Loading {detail}…")
            self.status.add_css_class("status-busy")
            self.loading_page.set_title(f"Loading {detail}")
            self.chat_stack.set_visible_child_name("loading")
        elif state == "ready":
            self.status.set_label(f"● {detail}")
            self.status.add_css_class("status-ready")
            self.show_chat()
            self.toast("Model loaded. The API is serving it too.")
        elif state == "stopping":
            self.status.set_label("Stopping…")
            self.status.add_css_class("status-busy")
        elif state == "failed":
            self.status.set_label("Load failed")
            self.status.add_css_class("status-failed")
            self.failed_page.set_description(detail)
            self.chat_stack.set_visible_child_name("failed")
        else:
            self.status.set_label("No model loaded")
            self.chat_stack.set_visible_child_name("empty")
            if self.web:
                self.web.load_uri("about:blank")
        self.update_controls()
        self.update_api_rows()
        self.update_banner()
        if self.closing and state in ("stopped", "failed"):
            self.close_now()
        return False

    def update_controls(self):
        state = self.engine.state
        self.load_button.set_label({"loading": "Cancel", "ready": "Eject", "stopping": "Stopping…"}.get(state, "Load"))
        self.load_button.set_sensitive(state != "stopping" and (state in ("loading", "ready") or bool(self.models)))
        if state in ("loading", "ready"):
            self.load_button.remove_css_class("suggested-action")
        else:
            self.load_button.add_css_class("suggested-action")
        self.picker.set_sensitive(state in ("stopped", "failed"))

    # ---------------------------------------------------------------- chat

    def show_chat(self):
        if not self.web:
            session = WebKit.NetworkSession.new(str(DATA_DIR / "web-data"), str(DATA_DIR / "web-cache"))
            self.content_manager = WebKit.UserContentManager()
            self.web = WebKit.WebView(network_session=session, user_content_manager=self.content_manager)
            self.web.connect("decide-policy", self.on_decide_policy)
            self.chat_stack.add_named(self.web, "web")
        # The chat app keeps its settings in localStorage; give it the server's API key before it starts.
        script = ("try{var k='LlamaCppWebui.config';var c=JSON.parse(localStorage.getItem(k)||'{}');"
                  f"c.apiKey={json.dumps(self.engine.settings.api_key)};"
                  "localStorage.setItem(k,JSON.stringify(c));}catch(e){}")
        self.content_manager.remove_all_scripts()
        self.content_manager.add_script(WebKit.UserScript.new(
            script, WebKit.UserContentInjectedFrames.TOP_FRAME, WebKit.UserScriptInjectionTime.START, None, None))
        self.web.load_uri(f"http://127.0.0.1:{self.engine.settings.port}/")
        self.chat_stack.set_visible_child_name("web")

    def on_decide_policy(self, _view, decision, kind):
        # Links out of the chat open in the default browser.
        if kind not in (WebKit.PolicyDecisionType.NAVIGATION_ACTION, WebKit.PolicyDecisionType.NEW_WINDOW_ACTION):
            return False
        action = decision.get_navigation_action()
        uri = action.get_request().get_uri()
        local = uri.startswith((f"http://127.0.0.1:{self.engine.settings.port}", "about:", "blob:", "data:"))
        clicked = action.get_navigation_type() == WebKit.NavigationType.LINK_CLICKED
        if not local and uri.startswith("http") and (clicked or kind == WebKit.PolicyDecisionType.NEW_WINDOW_ACTION):
            Gtk.UriLauncher.new(uri).launch(self, None, None, None)
            decision.ignore()
            return True
        return False

    # ---------------------------------------------------------------- serve

    def active_settings(self):
        return self.engine.settings if self.engine.state in ("loading", "ready") else self.settings

    def base_url(self):
        return f"http://127.0.0.1:{self.active_settings().port}/v1"

    def on_setting_changed(self, *_):
        s = self.settings
        s.port = int(self.port_row.get_value())
        s.context = int(self.context_row.get_value())
        s.memory_limit = self.limit_row.get_active()
        s.memory_gib = int(self.memory_row.get_value())
        s.threads = int(self.threads_row.get_value())
        s.prompt_cache = self.cache_row.get_active()
        s.api_key = self.key_row.get_text().strip()
        lan_changed = (s.lan, s.lan_port) != (self.lan_switch.get_active(), int(self.lan_port_row.get_value()))
        s.lan = self.lan_switch.get_active()
        s.lan_port = int(self.lan_port_row.get_value())
        s.save()
        self.memory_row.set_sensitive(s.memory_limit)
        if lan_changed and self.engine.state == "ready":
            # Network serving follows the switch at once, without a reload.
            self.engine.settings.lan, self.engine.settings.lan_port = s.lan, s.lan_port
            threading.Thread(target=self.engine.start_lan if s.lan else self.engine.stop_lan, daemon=True).start()
        self.update_api_rows()
        self.update_banner()

    def update_banner(self):
        running = self.engine.state in ("loading", "ready")
        self.banner.set_revealed(running and self.settings.server_key() != self.engine.settings.server_key())

    def update_api_rows(self):
        s = self.active_settings()
        state = self.engine.state
        model = self.engine.model if state in ("loading", "ready") else self.selected_model()
        model_id = model.name if model else "(none)"
        self.state_row.set_subtitle({
            "ready": "Serving", "loading": "Loading the model…", "stopping": "Stopping…",
            "failed": "The last load failed; see the log below",
        }.get(state, "Not serving: load a model to start the API"))
        self.url_row.set_subtitle(f"http://127.0.0.1:{s.port}/v1")
        self.model_row.set_subtitle(model_id)
        self.lan_row.set_visible(self.settings.lan)
        self.lan_row.set_subtitle(f"http://{lan_address()}:{self.settings.lan_port}/v1")
        auth = f' \\\n  -H "Authorization: Bearer {s.api_key}"' if s.api_key else ""
        key = s.api_key or "not-needed"
        self.example_buffer.set_text(
            f"curl http://127.0.0.1:{s.port}/v1/chat/completions \\\n"
            f'  -H "Content-Type: application/json"{auth} \\\n'
            f"  -d '{{\"model\": \"{model_id}\", \"messages\": [{{\"role\": \"user\", \"content\": \"Hello!\"}}]}}'\n\n"
            "# Python: pip install openai\n"
            "from openai import OpenAI\n"
            f'client = OpenAI(base_url="http://127.0.0.1:{s.port}/v1", api_key="{key}")\n'
            f'reply = client.chat.completions.create(model="{model_id}",\n'
            '    messages=[{"role": "user", "content": "Hello!"}])\n'
            "print(reply.choices[0].message.content)")

    # ---------------------------------------------------------------- log

    def queue_log(self, text):
        with self.log_lock:
            self.pending_log.append(text)

    def flush_log(self):
        with self.log_lock:
            text, self.pending_log = "".join(self.pending_log), []
        if text:
            adjustment = self.log_scroll.get_vadjustment()
            at_end = adjustment.get_value() + adjustment.get_page_size() >= adjustment.get_upper() - 40
            self.log_buffer.insert(self.log_buffer.get_end_iter(), text)
            excess = self.log_buffer.get_char_count() - LOG_LIMIT
            if excess > 0:
                self.log_buffer.delete(self.log_buffer.get_start_iter(), self.log_buffer.get_iter_at_offset(excess))
            if at_end:
                self.log_view.scroll_to_mark(self.log_end, 0, False, 0, 0)
        return True

    # ---------------------------------------------------------------- window

    def toast(self, text, timeout=3):
        self.toasts.add_toast(Adw.Toast(title=GLib.markup_escape_text(text), timeout=timeout))

    def on_open_run(self):
        folder = self.engine.run_dir or DATA_DIR / "runs"
        if folder.exists():
            Gtk.FileLauncher.new(Gio.File.new_for_path(str(folder))).launch(self, None, None, None)
        else:
            self.toast("No server has run yet")

    def on_about(self):
        about = Adw.AboutDialog(
            application_name="InferSSD Studio", application_icon=APP_ID, version=VERSION, developer_name="hammedrz",
            comments="Chat with and serve mixture-of-experts models whose experts stay on the SSD.\n\n"
                     "An unofficial app, not affiliated with WiCi AI, the authors of InferSSD. The Chat tab is "
                     "llama.cpp's chat app, served by inferssd.",
            website=WEBSITE, issue_url=WEBSITE + "/issues", license_type=Gtk.License.MIT_X11,
            copyright="© 2026 hammedrz")
        about.add_link("inferssd-gemma4 fork", "https://github.com/hammedrz/inferssd-gemma4")
        about.add_link("InferSSD (upstream)", "https://github.com/wici-ai/inferssd")
        about.present(self)

    def close_now(self):
        """Close the window even with a dialog open, which would otherwise take the close request for itself."""
        dialog = self.get_visible_dialog()
        if dialog:
            dialog.force_close()
        self.close()

    def on_close_request(self, _window):
        if self.engine.state in ("loading", "ready", "stopping"):
            # Stop the server first; on_engine_state closes the window once it has.
            if not self.closing:
                self.closing = True
                self.status.set_label("Stopping the server…")
                threading.Thread(target=self.engine.stop, daemon=True).start()
            return True
        return False


class App(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID, flags=Gio.ApplicationFlags.DEFAULT_FLAGS)
        self.window = None

    def do_startup(self):
        Adw.Application.do_startup(self)
        provider = Gtk.CssProvider()
        provider.load_from_string(CSS)
        Gtk.StyleContext.add_provider_for_display(Gdk.Display.get_default(), provider,
                                                  Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION)
        # The icon beside this file, for runs from a clone without install.sh.
        Gtk.IconTheme.get_for_display(Gdk.Display.get_default()).add_search_path(str(Path(__file__).resolve().parent))
        Gtk.Window.set_default_icon_name(APP_ID)
        for sig in (signal.SIGINT, signal.SIGTERM):
            GLibUnix.signal_add(GLib.PRIORITY_DEFAULT, sig, self.on_signal)
        self.set_accels_for_action("win.refresh", ["F5"])

    def do_activate(self):
        if not self.window:
            self.window = Window(self)
        self.window.present()

    def on_signal(self):
        if self.window:
            self.window.close_now()  # stops the server first
        else:
            self.quit()
        return True


if __name__ == "__main__":
    sys.exit(App().run(sys.argv))
