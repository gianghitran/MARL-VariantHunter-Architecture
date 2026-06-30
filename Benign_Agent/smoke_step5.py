"""Smoke test Bước 5: env dùng reference graphs benign-gen + malicious-gen."""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
from environment import ProvenanceGraphEnv

env = ProvenanceGraphEnv(gcn_model_path="trained_weights/darpa/gat.pth",
                         w2v_model_path="trained_weights/darpa/w2v.model",
                         run_dir="runs/_smoke5")
env.reset()
print("TESTLOG === live detect cycles (generate[+camo] -> detect) ===")
for c in range(12):
    env.step(0)            # Generation (Attack Agent) + camouflage selection
    _, _, _, info = env.step(2)   # Detection on (APT + camo)
    print(f"TESTLOG cyc{c} camo={info['camo_ratio']:.1f} conf={info['confidence_score']:.3f} "
          f"TP={info['TP']} FN={info['FN']} buf={info['replay_buf_size']} drift={info['drift']:.3f}")

print("TESTLOG === direct eval (reference benign-gen + malicious-gen + buffer) ===")
f1 = env.evaluate_detection_mlp(run_dir="runs/_smoke5")
print(f"TESTLOG eval_global_f1={f1}")
print(f"TESTLOG buffer_final={len(env.replay_buffer)}")

if len(env.replay_buffer) >= 2:
    print("TESTLOG === EWC update (buffer has FN samples) ===")
    f1b = env.train_detection_agent_ewc(run_dir="runs/_smoke5", batch_size=8)
    print(f"TESTLOG ewc_global_f1={f1b}")
else:
    print("TESTLOG EWC skipped (buffer < 2, FN=0 trong smoke) — kiem tra eval o tren")
print("TESTLOG SMOKE5 DONE")
