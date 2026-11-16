"""The layout model: Qwen3.5 9B (MLX build), run with mlx-lm.
Callers in thread/.venv use ask(); it runs this file in spike/.venv, which has MLX, one process per call, so the model
leaves memory as soon as it has answered (one large model at a time). setup.sh downloads the weights; at run time
nothing goes to the network."""
import sys, os, json, subprocess, time

MODEL, REVISION = "mlx-community/Qwen3.5-9B-4bit", "8b2b98c00a6b4d291155e4890773ca8f769aee53"
HERE = os.path.dirname(os.path.abspath(__file__))
PY_IMG = os.path.join(HERE, "../spike/.venv/bin/python")

class ModelError(OSError): pass      # an OSError, like the network errors the callers already handle

def ask(prompt, max_tokens=4096, temperature=0.1, timeout=600):
    """-> (the model's text, seconds taken). Raises ModelError, or TimeoutError if it takes longer than `timeout`."""
    t = time.time()
    try:
        r = subprocess.run([PY_IMG, __file__], input=json.dumps({"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired: raise TimeoutError("the layout model took too long")
    except OSError as e: raise ModelError(f"could not start the layout model: {e}")
    if r.returncode: raise ModelError((r.stderr or r.stdout)[-1500:].strip() or "the layout model stopped")
    return r.stdout, time.time() - t

def json_in(txt):           # the answer as JSON: the model sometimes wraps it in words or a code fence
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"                  # the weights are on disk already; never call out
    q = json.load(sys.stdin)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    try: model, tok = load(MODEL, revision=REVISION)
    except Exception as e: sys.exit(f"The layout model is not downloaded. Open the app again to finish setup, or run ./setup.sh. ({type(e).__name__})")
    text = tok.apply_chat_template([{"role": "user", "content": q["prompt"]}], add_generation_prompt=True, tokenize=False, enable_thinking=False)
    sys.stdout.write(generate(model, tok, text, max_tokens=q["max_tokens"], sampler=make_sampler(temp=q["temperature"])))
