# Running on Google Colab (T4 GPU)

Runtime → Change runtime type → T4 GPU. Then run these cells in order.

```python
# 1. Get the code
!git clone https://github.com/vaibhintel/rf-data-agent.git
%cd rf-data-agent
!pip install -q -e ".[dev]"
```

```python
# 2. Install and start Ollama, keeping model weights on Drive so they persist across sessions
from google.colab import drive
drive.mount('/content/drive')

!apt-get -qq install -y zstd > /dev/null
!curl -fsSL https://ollama.com/install.sh | sh

import os, subprocess, time, requests
OLLAMA_MODELS = '/content/drive/MyDrive/rf_agent_project/ollama_models'
os.makedirs(OLLAMA_MODELS, exist_ok=True)
env = dict(os.environ, OLLAMA_MODELS=OLLAMA_MODELS,
           OLLAMA_CONTEXT_LENGTH='16384', OLLAMA_KEEP_ALIVE='60m')
subprocess.Popen(['ollama', 'serve'], env=env,
                 stdout=open('/content/ollama.log', 'w'), stderr=subprocess.STDOUT)
for _ in range(30):
    try:
        requests.get('http://localhost:11434/api/version', timeout=1); break
    except requests.exceptions.ConnectionError:
        time.sleep(1)

!OLLAMA_MODELS={OLLAMA_MODELS} ollama pull qwen2.5:7b
```

```python
# 3. Tests (no model needed), then the agent and the evals
!pytest -q
!rf-agent --trace "Did any firmware build make ACLR worse? Which build, and at what temperature?"
!python evals/run_evals.py --repeats 3 --out evals/results/qwen2.5-7b.json
```

Or from a notebook cell directly (Colab supports top-level `await`):

```python
from rf_agent.cli import build_agent
res = await build_agent().run("What is the pass rate for 25.3.0 at 60 C?")
print(res.answer)
for s in res.trace:
    print(s.step, s.tool, s.arguments, "ok" if s.ok else "ERROR")
```
