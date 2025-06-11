import modal

# needed for installing from nightlies
VLLM_COMMIT = "4f5b059f146adeecd153fa781cf21863ed6679d8"

vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"https://wheels.vllm.ai/{VLLM_COMMIT}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl",
        "huggingface_hub[hf_transfer]==0.26.2",
        "flashinfer-python==0.2.0.post2",  # pin tightly!
        extra_index_url="https://flashinfer.ai/whl/cu124/torch2.5",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

vllm_image = vllm_image.env({"VLLM_USE_V1": "0"})

MODELS_DIR = "/llamas"
MODEL_NAME = "charlesfrye/Ministral-8B-Instruct-2410-FP8-Dynamic"
MODEL_REVISION = "d24575707780d80a226e2a71226f637ecde6d63b"


hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("example-vllm-mistral")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H100:{N_GPU}",
    scaledown_window=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(
    max_inputs=20,  # 2000 total sequences
)
@modal.web_server(port=VLLM_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "vllm",
        "serve",
        "--uvicorn-log-level=info",
        MODEL_NAME,
        "--revision",
        MODEL_REVISION,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        # application-specific config
        "--disable-sliding-window",  # shorter prompts, not compatible with chunked prefill
        "--max-model-len",  # prompts are short
        str(2 << 10),
        "--enable-prefix-caching",  # turn on prefix caching
        "--disable-log-requests",  # don't log individual requests, we make too many
    ]

    subprocess.Popen(" ".join(cmd), shell=True)


@app.local_entrypoint()
def test(test_timeout=5 * MINUTES):
    import json
    import time
    import urllib

    print(f"Running health check for server at {serve.get_web_url()}")
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(serve.get_web_url() + "/health") as response:
                if response.getcode() == 200:
                    up = True
        except Exception:
            if time.time() - start > test_timeout:
                break
            time.sleep(delay)

    assert up, f"Failed health check for server at {serve.get_web_url()}"

    print(f"Successful health check for server at {serve.get_web_url()}")

    messages = [
        {
            "role": "system",
            "content": "Réponds exclusivement en français, jamais en anglais, en commençant chaque message par un emoji 🥖.",
        }
    ]
    messages.append({"role": "user", "content": "Testing! Is this thing on?"})
    print(f"Sending a sample message to {serve.get_web_url()}", *messages, sep="\n")

    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"messages": messages, "model": MODEL_NAME})
    req = urllib.request.Request(
        serve.get_web_url() + "/v1/chat/completions",
        data=payload.encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req) as response:
        print(json.loads(response.read().decode()))
