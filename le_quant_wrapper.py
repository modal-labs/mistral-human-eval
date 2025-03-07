import modal

app = modal.App("example-quantize-mistral")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("llmcompressor==0.4.1")
    .add_local_python_source("le_quant")
)

volume = modal.Volume.from_name("huggingface-cache", create_if_missing=True)

MINUTES = 60  # seconds


@app.function(
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=image,
    gpu="h100",
    volumes={"/root/.cache/huggingface": volume},
    timeout=30 * MINUTES,
)
def go():
    import le_quant  # noqa

    le_quant.run()
