
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

APP = PROJECT / "banking_app.py"
BACKEND = PROJECT / "banking_app_backend.py"
RUNTIME = PROJECT / "banking_assistant_runtime.py"

OLLAMA_EXE = (
    Path(os.environ.get("LOCALAPPDATA", ""))
    / "Programs" / "Ollama" / "ollama.exe"
)

MODEL = "banking-assistant:final"
PORT = 8503
URL = f"http://localhost:{PORT}"

def api_get(path, timeout=3):

    with urllib.request.urlopen(
        "http://127.0.0.1:11434" + path,
        timeout=timeout
    ) as response:

        return json.loads(
            response.read().decode("utf-8")
        )


def ollama_ready():

    try:
        api_get("/api/tags")
        return True
    except Exception:
        return False


def start_ollama():

    if ollama_ready():
        print("Ollama server is already running.")
        return True

    if not OLLAMA_EXE.exists():
        print("Ollama executable not found:", OLLAMA_EXE)
        return False

    print("Starting Ollama server...")

    log_file = PROJECT / "ollama_server.log"

    log_handle = open(
        log_file,
        "a",
        encoding="utf-8"
    )

    subprocess.Popen(
        [str(OLLAMA_EXE), "serve"],
        cwd=str(PROJECT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=getattr(
            subprocess,
            "CREATE_NO_WINDOW",
            0
        )
    )

    for _ in range(15):

        time.sleep(1)

        if ollama_ready():
            print("Ollama server started.")
            return True

    print("Ollama API did not become ready.")
    print("Check:", log_file)
    return False


def ensure_model():

    try:

        data = api_get("/api/tags")

        names = [
            item.get("name", "")
            for item in data.get("models", [])
        ]

        if any(
            name == MODEL or name.startswith(MODEL + ":")
            for name in names
        ):
            print("Banking assistant Ollama model found.")
            return True

    except Exception as error:
        print("Model listing error:", error)
        return False

    if not OLLAMA_EXE.exists() or not (PROJECT / "ollama_export" / "final_banking_assistant" / "Modelfile").exists():
        print("Final Ollama model or Modelfile is missing.")
        return False

    print("Creating final Ollama model...")

    result = subprocess.run(
        [
            str(OLLAMA_EXE),
            "create",
            MODEL,
            "-f",
            str(PROJECT / "ollama_export" / "final_banking_assistant" / "Modelfile")
        ],
        cwd=str(PROJECT),
        text=True
    )

    return result.returncode == 0


def port_ready():

    try:

        with urllib.request.urlopen(
            URL + "/_stcore/health",
            timeout=2
        ) as response:

            return response.status == 200

    except Exception:
        return False


def main():

    print("=" * 60)
    print("BANKING AND CREDIT RISK ASSISTANT")
    print("=" * 60)

    for path in [APP, BACKEND, RUNTIME]:

        if not path.exists():
            print("Required file missing:", path)
            input("Press Enter to exit...")
            return 1

    manifest = (
        PROJECT / "data_knowledge"
        / "lendingclub_full_v1" / "manifest.json"
    )

    if not manifest.exists():
        print("LendingClub dataset manifest is missing.")
        input("Press Enter to exit...")
        return 1

    if not start_ollama():
        input("Start Ollama manually, then press Enter...")
        if not ollama_ready():
            return 1

    if not ensure_model():
        print("Banking assistant model unavailable.")
        input("Press Enter to exit...")
        return 1

    if port_ready():

        print("Dashboard already running:", URL)
        webbrowser.open(URL)
        input("Press Enter to close this launcher...")
        return 0

    print("\nStarting Streamlit dashboard...")
    print("URL:", URL)

    webbrowser.open(URL)

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.port",
        str(PORT),
        "--server.headless",
        "true"
    ]

    try:
        return subprocess.call(
            command,
            cwd=str(PROJECT)
        )

    except KeyboardInterrupt:
        print("\nAssistant stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
