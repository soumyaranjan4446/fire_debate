import sys
import os
import torch

# --- FIX: Add Project Root to Path ---
# This tells Python to look one level up for the 'fire_debate' package
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("🔍 FIRE-DEBATE SYSTEM DIAGNOSTIC")
print("================================")
print(f"📂 Project Root Detected: {project_root}")

# 1. Check Imports
print("[1/5] Checking Imports...")
try:
    from fire_debate.insight.fallacy import FallacyDetector
    from fire_debate.insight.graph_builder import GraphBuilder
    from fire_debate.insight.hgt_judge import HGTJudge
    from fire_debate.schemas.debate import DebateLog, DebateTurn
    print("   ✅ Imports Successful.")
except ImportError as e:
    print(f"   ❌ IMPORT ERROR: {e}")
    print("      (Make sure you are running this from the project root!)")
    sys.exit(1)

# 2. Check Fallacy Detector (Relevance Mode)
print("\n[2/5] Testing Fallacy Detector (Logic + Relevance)...")
try:
    # Force CPU for quick test to avoid VRAM initialization overhead
    fd = FallacyDetector(device="cpu") 
    claim = "The earth is flat."
    arg_bad = "I like ice cream."
    arg_good = "Satellite imagery confirms the curvature of the Earth."
    
    score_bad = fd.detect(arg_bad, context=claim)
    score_good = fd.detect(arg_good, context=claim)
    
    print(f"   Context: '{claim}'")
    print(f"   - Arg: '{arg_bad}' -> Rel: {score_bad.get('relevance', 0):.2f}")
    print(f"   - Arg: '{arg_good}' -> Rel: {score_good.get('relevance', 0):.2f}")
    
    if score_good['relevance'] > score_bad['relevance']:
        print("   ✅ Relevance Filter Working.")
    else:
        print("   ⚠️ WARNING: Relevance filter inconclusive (Check NLI model).")
except Exception as e:
    print(f"   ❌ FALLACY DETECTOR FAILED: {e}")

# 3. Check Graph Builder (Context Injection)
print("\n[3/5] Testing Graph Builder (Context Injection)...")
try:
    # Create Dummy Log
    log = DebateLog(debate_id="test", claim_id="0", claim_text="Test Claim", ground_truth=False)
    log.add_turn(DebateTurn(turn_id="1", agent_id="A", stance="PRO", phase="OPEN", text="Test Argument", citations=[]))
    
    # Force CPU for builder test
    builder = GraphBuilder(device="cpu")
    data = builder.build_graph(log)
    
    # Check Feature Size: 384(Arg) + 384(Context) + 1(Stance) + 1(Logic) + 1(Rel) = 771
    feat_dim = data['argument'].x.shape[1]
    print(f"   Feature Dimension: {feat_dim}")
    
    if feat_dim == 771:
        print("   ✅ Context Injection Verified (771 dims).")
    else:
        print(f"   ❌ DIMENSION MISMATCH: Expected 771, got {feat_dim}")
except Exception as e:
    print(f"   ❌ GRAPH BUILDER FAILED: {e}")

# 4. Check HGT Judge (Attention Pooling)
print("\n[4/5] Testing HGT Judge (Forward Pass)...")
try:
    # Force CPU for judge test
    judge = HGTJudge(device="cpu")
    result = judge.judge(log)
    print(f"   Verdict: {result['verdict']} (Conf: {result['confidence']:.2f})")
    print("   ✅ GNN Forward Pass Successful.")
except Exception as e:
    print(f"   ❌ JUDGE FAILED: {e}")

print("\n================================")
print("🟢 SYSTEM READY FOR DEBATE")