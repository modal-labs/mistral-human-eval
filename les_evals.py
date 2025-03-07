import time
from pathlib import Path

import modal

app = modal.App("humaneval-sandbox")

volume = modal.Volume.from_name("mistral-humaneval", create_if_missing=False)

sandbox_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/modal-labs/human-eval.git",
        "pip install -e human-eval",
    )
)

MINUTES = 60


@app.function(volumes={"/humaneval": volume}, timeout=10 * MINUTES)
def run_humaneval(sample_file_path: str, problem_file_path: str):
    volume.reload()

    with modal.Volume.ephemeral() as sandboxed_volume:
        with sandboxed_volume.batch_upload() as batch:
            batch.put_file(sample_file_path, "samples.jsonl")
            batch.put_file(problem_file_path, "problems.jsonl")

        print(f"Starting sandbox for {sample_file_path}")
        sandbox = modal.Sandbox.create(
            "bash",
            "-c",
            "evaluate_functional_correctness vol/samples.jsonl --problem_file=vol/problems.jsonl --n_workers=32",
            image=sandbox_image,
            volumes={"/vol": sandboxed_volume},
            timeout=5 * MINUTES,
            cpu=32,
        )

        try:
            start = time.time()
            while (time.time() - start) < 5 * MINUTES:
                if sandbox.poll() is not None:
                    print(f"Finished sandbox for {sample_file_path}")
                    break
                else:
                    time.sleep(1)
            else:
                raise TimeoutError
        except TimeoutError:
            print("Sandbox timed out")

        try:
            data = b""
            for chunk in sandboxed_volume.read_file("samples.jsonl_results.jsonl"):
                data += chunk
        except Exception:
            pass

        Path(f"{sample_file_path}_results.jsonl").write_bytes(data)


@app.function(volumes={"/humaneval": volume}, timeout=30 * MINUTES)
def compute_results():
    import os

    envs = [element for element in Path("/humaneval").iterdir() if element.is_dir()]
    envs = list(filter(lambda p: not p.stem.startswith("."), envs))
    volume.reload()

    for env in envs:
        done = False
        while not done:
            print(f"looking in {env}")
            problem_file = env / "data.jsonl"

            pattern = "*/*.jsonl"
            handles = []
            for file_path in env.glob(pattern):
                # Skip results files
                if str(file_path).endswith("_results.jsonl"):
                    continue

                # Check if the corresponding results file exists
                results_file = f"{file_path}_results.jsonl"
                if not os.path.exists(results_file):
                    # If it doesn't exist, run run_humaneval
                    print(f"Checking {file_path}")
                    handles.append(run_humaneval.spawn(file_path, problem_file))

            if not handles:
                print(f"no new files in {env}")
                done = True
            else:
                print(f"{len(handles)} new files in {env}, repeating")

                modal.FunctionCall.gather(*handles)

            volume.reload()


@app.local_entrypoint()
def main():
    compute_results.remote()
