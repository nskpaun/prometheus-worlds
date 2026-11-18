"""The world-plan model: Qwen3.5 35B-A3B (MLX build, 4 bit), run with mlx-lm. It writes the list of things, the style
guide and the material choices. setup.sh downloads the weights; at run time nothing goes to the network.
Callers in thread/.venv use Chat (a conversation) or ask_many (several questions, one load). Both run this file in
spike/.venv, which has MLX, as ONE process that leaves memory when its last answer is in: one large model at a time.
  spike/.venv/bin/python chat.py      one JSON a line in: {"messages": [{"role", "content"}...], "max_tokens", "temperature"}
                                      one JSON a line out: {"text", "s", "gen_tokens", "peak_gb", "finish"}; the first line out is {"load_s"}"""
import sys, os, json, subprocess, time
import where

MODEL, REVISION = "mlx-community/Qwen3.5-35B-A3B-4bit", "1e20fd8d42056f870933bf98ca6211024744f7ec"

class Chat:
    """One model process held for a conversation: `with Chat() as c: c.say(messages)`."""
    def __init__(self):
        t = time.time()
        self.p = subprocess.Popen([where.PY_IMG, "-B", os.path.abspath(__file__)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        line = self.p.stdout.readline()
        if not line: raise SystemExit("The world-plan model could not be started. Open the app again to finish setup, or run ./setup.sh.")
        self.load_s = json.loads(line)["load_s"]
    def say(self, messages, max_tokens=5200, temperature=0.1):
        self.p.stdin.write(json.dumps({"messages": messages, "max_tokens": max_tokens, "temperature": temperature}) + "\n"); self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line: raise SystemExit("The world-plan model stopped before it had answered.")
        return json.loads(line)
    def close(self):
        try: self.p.stdin.close(); self.p.wait(timeout=60)
        except Exception: self.p.kill()
    def __enter__(self): return self
    def __exit__(self, *a): self.close()

def ask_many(prompts, max_tokens=3000, temperature=0.2):
    """Several questions, each on its own, with one load of the model. "{{GUIDE}}" in a prompt is replaced by the answer
    to the last prompt that stood without it. -> ([text, ...], {"load_s", "s": [...], "wall_s"})"""
    t = time.time(); out, secs, guide = [], [], ""
    with Chat() as c:
        for p in prompts:
            own = "{{GUIDE}}" not in p
            x = c.say([{"role": "user", "content": p.replace("{{GUIDE}}", guide)}], max_tokens, temperature); out.append(x["text"]); secs.append(x["s"])
            if own: guide = x["text"]
        load = c.load_s
    return out, {"load_s": load, "s": secs, "wall_s": round(time.time() - t, 1)}

if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"                  # the weights are on disk already; never call out
    import mlx.core as mx
    mx.set_cache_limit(2 * 2**30)
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    t = time.time()
    try: model, tok = load(MODEL, revision=REVISION)
    except Exception as e: sys.exit(f"The world-plan model could not be loaded ({type(e).__name__}: {str(e)[:200]}). If it is not downloaded, open the app again to finish setup, or run ./setup.sh.")
    print(json.dumps({"load_s": round(time.time() - t, 1)}), flush=True)
    for line in sys.stdin:
        if not line.strip(): continue
        q = json.loads(line)
        text = tok.apply_chat_template(q["messages"], add_generation_prompt=True, tokenize=False, enable_thinking=False)
        t = time.time(); parts, last = [], None
        for r in stream_generate(model, tok, text, max_tokens=q.get("max_tokens", 5000), sampler=make_sampler(temp=q.get("temperature", 0.1))):
            parts.append(r.text); last = r
        print(json.dumps({"text": "".join(parts), "s": round(time.time() - t, 1), "gen_tokens": last.generation_tokens, "peak_gb": round(last.peak_memory, 2), "finish": last.finish_reason}), flush=True)
