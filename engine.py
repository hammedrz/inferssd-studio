# SPDX-License-Identifier: MIT
"""The inferssd server behind InferSSD Studio: finding inferssd and models, settings, and one owned server.

Nothing here touches GTK. The window drives an Engine and receives its events through callbacks,
which run on worker threads; the window hands them to the GTK main loop.
"""

import json
import os
import signal
import socket
import struct
import subprocess
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Where inferssd's installer puts it: releases, the `current` link to the active one, and downloaded models.
INFERSSD_HOME = Path(os.environ.get("INFERSSD_HOME", Path.home() / ".local/share/inferssd"))
CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "inferssd-studio"
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "inferssd-studio"

# The GGUF architectures whose routed experts inferssd reads from the SSD. deepseek2 and gemma4 need the
# hammedrz/inferssd-gemma4 fork; upstream inferssd runs the others.
SUPPORTED_ARCHITECTURES = {"deepseek2", "gemma4", "deepseek41", "muse-spark", "qwen4exp"}
SKIPPED_PREFIXES = ("mmproj", "mtp-", "dflash")


@dataclass(frozen=True)
class Installation:
    root: Path
    launcher: Path
    resources: Path


def find_inferssd(chosen=""):
    """The inferssd to run, from the folder chosen in the window, $INFERSSD_DIR, or the installer's
    INFERSSD_HOME/current; None when none of them holds one.

    A folder qualifies when it is an installer release (venv/bin/inferssd) or a source checkout set up as
    inferssd's quick start describes (.venv/bin/inferssd), with its engine-resources.json beside either.
    """
    for folder in (chosen, os.environ.get("INFERSSD_DIR"), INFERSSD_HOME / "current"):
        if folder and (installation := inferssd_at(folder)):
            return installation
    return None


def inferssd_at(folder):
    """The inferssd installation in FOLDER itself, or None."""
    root = Path(folder).expanduser()
    resources = root / "engine-resources.json"
    for launcher in (root / "venv/bin/inferssd", root / ".venv/bin/inferssd"):
        if launcher.is_file() and resources.is_file():
            # For trying the window without a model: a stand-in with the launcher's command line.
            stand_in = os.environ.get("INFERSSD_STUDIO_LAUNCHER")
            return Installation(root, Path(stand_in) if stand_in else launcher, resources)
    return None


@dataclass
class Settings:
    inferssd_dir: str = ""  # empty: find it (see find_inferssd)
    model_dirs: list = field(default_factory=lambda: [str(INFERSSD_HOME / "models")])
    model: str = ""
    port: int = 8080
    context: int = 8192
    memory_limit: bool = False
    memory_gib: int = 7
    threads: int = 0  # 0: the launcher chooses for this machine
    prompt_cache: bool = False
    api_key: str = ""
    lan: bool = False
    lan_port: int = 1234

    @classmethod
    def load(cls):
        try:
            values = json.loads((CONFIG_DIR / "settings.json").read_text())
            known = {k: v for k, v in values.items() if k in cls.__dataclass_fields__}
            return cls(**known)
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIG_DIR / "settings.json"
        path.with_suffix(".tmp").write_text(json.dumps(asdict(self), indent=2))
        path.with_suffix(".tmp").replace(path)

    def server_key(self):
        """The settings a running server was started with; others apply at once."""
        return (self.model, self.port, self.context, self.memory_limit and self.memory_gib, self.threads,
                self.prompt_cache, self.api_key)


@dataclass
class Model:
    path: str
    name: str
    size: int
    architecture: str
    experts: int

    @property
    def supported(self):
        # An architecture can have dense models too (Gemma 4 E4B beside 26B-A4B): only routed experts stream.
        return self.architecture in SUPPORTED_ARCHITECTURES and self.experts > 0


def gguf_info(path):
    """(general.architecture, ARCH.expert_count) from a GGUF header; ('', 0) where absent."""
    architecture, experts = "", 0
    try:
        with open(path, "rb") as f:
            def take(fmt):
                data = f.read(struct.calcsize(fmt))
                return struct.unpack("<" + fmt, data)[0]

            def string():
                return f.read(take("Q")).decode("utf-8", "replace")

            if f.read(4) != b"GGUF":
                return architecture, experts
            take("I"), take("Q")
            scalar = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
            integer = {4: "I", 5: "i", 10: "Q", 11: "q"}
            for _ in range(take("Q")):
                key, kind = string(), take("I")
                if key == "general.architecture" and kind == 8:
                    architecture = string()
                elif architecture and key == f"{architecture}.expert_count" and kind in integer:
                    return architecture, take(integer[kind])
                elif kind in scalar:
                    f.seek(scalar[kind], 1)
                elif kind == 8:
                    f.seek(take("Q"), 1)
                elif kind == 9:
                    element, count = take("I"), take("Q")
                    if element in scalar:
                        f.seek(scalar[element] * count, 1)
                    else:
                        for _ in range(count):
                            f.seek(take("Q"), 1)
                else:
                    break
    except (OSError, struct.error):
        pass
    return architecture, experts


def find_models(folders):
    """Every GGUF model under FOLDERS: one entry per model, its first shard, the size of all its shards."""
    models = []
    for folder in folders:
        for path in sorted(Path(folder).expanduser().rglob("*.gguf")):
            name = path.name
            if name.startswith(SKIPPED_PREFIXES) or "-of-" in name and "-00001-of-" not in name:
                continue
            size = path.stat().st_size
            if "-00001-of-" in name:
                prefix = name.split("-00001-of-")[0]
                size = sum(p.stat().st_size for p in path.parent.glob(prefix + "-*-of-*.gguf"))
                name = prefix
            models.append(Model(str(path), name.removesuffix(".gguf"), size, *gguf_info(path)))
    return models


def port_free(port, host="127.0.0.1"):
    with socket.socket() as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def lan_address():
    """This machine's address on its local network, as other devices reach it.

    The main routing table's default route gives it (its source, or its device's address). Asking the
    kernel which route a packet would take does not: a VPN or a container bridge can claim that route.
    """
    def run(*argv):
        return subprocess.run(argv, capture_output=True, text=True, timeout=2).stdout.split()

    try:
        words = run("ip", "-4", "route", "show", "default")
        if "src" in words:
            return words[words.index("src") + 1]
        if "dev" in words:
            address = run("ip", "-4", "-o", "addr", "show", "dev", words[words.index("dev") + 1])
            if "inet" in address:
                return address[address.index("inet") + 1].split("/")[0]
    except (OSError, subprocess.SubprocessError):
        pass
    return socket.gethostbyname(socket.gethostname())


class Forwarder:
    """Plain TCP forwarding from 0.0.0.0:PORT to the loopback server, for other devices on the network."""

    def __init__(self, listen_port, target_port):
        self.target = ("127.0.0.1", target_port)
        self.server = socket.create_server(("0.0.0.0", listen_port), reuse_port=False)
        self.server.settimeout(0.5)
        self.stopped = threading.Event()
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self.stopped.is_set():
            try:
                client, _ = self.server.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            try:
                upstream = socket.create_connection(self.target, timeout=5)
            except OSError:
                client.close()
                continue
            upstream.settimeout(None)
            for a, b in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    @staticmethod
    def _pump(source, destination):
        try:
            while data := source.recv(65536):
                destination.sendall(data)
        except OSError:
            pass
        finally:
            for s in (source, destination):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self):
        self.stopped.set()
        self.server.close()


class Engine:
    """One inferssd server at a time, started with `inferssd launch` and stopped with Ctrl+C as a person would.

    States: stopped, loading, ready, stopping, failed. on_state(state, detail) and on_log(text) are called
    from worker threads.
    """

    def __init__(self, on_state, on_log):
        self.on_state, self.on_log = on_state, on_log
        self.state = "stopped"
        self.process = None
        self.settings = None  # the settings the current server was started with
        self.model = None
        self.run_dir = None
        self.forwarder = None
        self.lock = threading.Lock()
        self.generation = 0
        self.launcher_log = None

    def _log(self, text):
        """The launcher's output and Studio's own messages: to the window and beside the run's folder."""
        if self.launcher_log:
            try:
                self.launcher_log.write(text)
            except (OSError, ValueError):
                pass
        self.on_log(text)

    def _set(self, state, detail=""):
        self.state = state
        self.on_state(state, detail)

    def start(self, model, settings):
        with self.lock:
            if self.state not in ("stopped", "failed"):
                raise RuntimeError("a model is already loaded")
            installation = find_inferssd(settings.inferssd_dir)
            if not installation:
                raise RuntimeError("inferssd was not found: install it, or choose its folder on the Serve tab")
            if not port_free(settings.port):
                raise RuntimeError(f"port {settings.port} is already in use; choose another on the Serve tab")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            memory = settings.memory_gib * 2**30 if settings.memory_limit else (
                os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
            policy = DATA_DIR / "runtime-policy.json"
            policy.write_text(json.dumps({
                "platform": "linux", "lock_path": str(DATA_DIR / "runtime.lock"),
                "stop_path": str(DATA_DIR / "runtime.stop"), "memory_budget_bytes": memory,
                "minimum_memlock_bytes": 0, "cleanup_seconds": 30,
            }))
            runs = DATA_DIR / "runs"
            runs.mkdir(exist_ok=True)
            self.run_dir = runs / time.strftime("%Y%m%d-%H%M%S")
            if self.launcher_log:
                self.launcher_log.close()
            # The launcher creates the run folder itself (it must be new), so this log sits beside it.
            self.launcher_log = open(self.run_dir.with_suffix(".launcher.log"), "a", buffering=1, errors="replace")
            argv = [str(installation.launcher), "launch", "--execute", "--resources", str(installation.resources),
                    "--model", model.path,
                    "--runtime-policy", str(policy), "--port", str(settings.port),
                    "--context", str(settings.context), "--output-dir", str(self.run_dir)]
            if settings.threads > 0:
                argv += ["--threads", str(settings.threads), "--batch-threads", str(settings.threads)]
            if not settings.prompt_cache:
                argv += ["--prompt-cache", "off"]
            argv += ["--", "--alias", model.name]
            if settings.api_key:
                argv += ["--api-key", settings.api_key]
            self._log(f"$ {' '.join(a if a != settings.api_key else '••••' for a in argv)}\n")
            self.process = subprocess.Popen(argv, cwd=str(installation.root), stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                            start_new_session=True, text=True, errors="replace")
            self.settings = Settings(**asdict(settings))
            self.model = model
            self.generation += 1
            generation = self.generation
            self._set("loading", model.name)
        threading.Thread(target=self._read_launcher, args=(self.process,), daemon=True).start()
        threading.Thread(target=self._tail_server_log, args=(generation,), daemon=True).start()
        threading.Thread(target=self._wait_ready, args=(self.process, generation), daemon=True).start()

    def _read_launcher(self, process):
        for line in process.stdout:
            # The launcher's receipts are long JSON lines; keep the log readable.
            self._log(line if len(line) < 400 else line[:400] + " …\n")

    def _tail_server_log(self, generation):
        path, offset = None, 0
        while generation == self.generation and self.state in ("loading", "ready", "stopping"):
            path = path or (self.run_dir / "server.log" if (self.run_dir / "server.log").exists() else None)
            if path:
                try:
                    with open(path, errors="replace") as f:
                        f.seek(offset)
                        text = f.read()
                        offset = f.tell()
                    if text:
                        self.on_log(text)
                except OSError:
                    pass
            time.sleep(0.5)

    def _wait_ready(self, process, generation):
        url = f"http://127.0.0.1:{self.settings.port}/health"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while process.poll() is None and generation == self.generation:
            try:
                with opener.open(url, timeout=2) as response:
                    if response.status == 200 and self.state == "loading":
                        if self.settings.lan:
                            self.start_lan()
                        self._set("ready", self.model.name)
            except OSError:
                pass
            time.sleep(1 if self.state == "loading" else 2)
        code = process.wait()
        # Whichever of this thread and stop() sees the exit first settles the state, once.
        with self.lock:
            if generation != self.generation:
                return
            self.generation += 1
            self.stop_lan()
            if self.state == "stopping":
                self._set("stopped")
            else:
                self._set("failed", f"the server exited with code {code}; see the log on the Serve tab")

    def stop(self):
        """Stop the server and wait for its cleanup; the state is final on return. Blocking: call it from a
        worker thread."""
        with self.lock:
            process, generation = self.process, self.generation
            if not process or process.poll() is not None:
                self.generation += 1
                self._set("stopped")
                return
            self._set("stopping")
        self.stop_lan()
        try:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        except ProcessLookupError:
            process.wait()
        with self.lock:
            if generation == self.generation:
                self.generation += 1
                self._set("stopped")

    def start_lan(self):
        self.stop_lan()
        try:
            self.forwarder = Forwarder(self.settings.lan_port, self.settings.port)
            self._log(f"Serving on the local network at http://{lan_address()}:{self.settings.lan_port}/v1\n")
        except OSError as error:
            self._log(f"Local network serving failed on port {self.settings.lan_port}: {error}\n")

    def stop_lan(self):
        if self.forwarder:
            self.forwarder.close()
            self.forwarder = None

    def server_log_path(self):
        if not self.run_dir:
            return None
        for name in ("server.log", "server.log.gz"):
            if (self.run_dir / name).exists():
                return self.run_dir / name
        return None
