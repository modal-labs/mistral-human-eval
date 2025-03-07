MODEL_ID = "mistralai/Ministral-8B-Instruct-2410"
MODEL_REVISION = "4847e87e5975a573a2a190399ca62cd266c899ad"


def run(
    model_id=MODEL_ID,
    model_revision=MODEL_REVISION,
    scheme="FP8_DYNAMIC",
    check=True,
):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.transformers import oneshot

    print(f"🥖 quantizing {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=model_revision, device_map="auto", torch_dtype="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"]
    )

    oneshot(model=model, recipe=recipe)

    if check:
        print("🥖" * 10 + " SAMPLE GENERATION " + "🥖" * 10)
        input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(
            "cuda"
        )
        output = model.generate(input_ids, max_new_tokens=20)
        print(tokenizer.decode(output[0]))
        print("🥖" * 30)

    model_name = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
    model.push_to_hub(model_name)
    tokenizer.push_to_hub(model_name)
