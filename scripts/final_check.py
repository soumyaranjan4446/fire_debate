import sys
import os
import torch

# --- 1. ROBUST PATH SETUP ---
script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(script_path)
package_dir = os.path.dirname(scripts_dir)
project_root = os.path.dirname(package_dir)

print(f"🔧 Force-adding project root to Python Path: {project_root}")
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 2. IMPORTS (Updated for your Schema) ---
try:
    from fire_debate.insight.hgt_judge import HGTJudge
    # IMPORT CHANGE: Use your correct class names
    from fire_debate.schemas.debate import DebateLog, DebateTurn 
    print("✅ Successfully imported fire_debate modules.")
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def test_pipeline():
    print("\n🚀 Starting System Integrity Check...")
    
    # 1. Mock Data (Using your Dataclass Structure)
    dummy_log = DebateLog(
        debate_id="test_001",
        claim_id="claim_001",
        claim_text="AI will replace doctors in the next 10 years.",
        ground_truth=None,
        turns=[
            DebateTurn(
                turn_id="t1",
                agent_id="ag1",
                agent_name="Alice",
                stance="PRO",
                text="AI is statistically more accurate at diagnostics than humans.",
                round=1,
                phase="ARGUMENT"
            ),
            DebateTurn(
                turn_id="t2",
                agent_id="ag2",
                agent_name="Bob",
                stance="CON",
                text="But AI lacks the empathy required for patient care and ethics.",
                round=1,
                phase="REBUTTAL"
            ),
            DebateTurn(
                turn_id="t3",
                agent_id="ag1",
                agent_name="Alice",
                stance="PRO",
                text="Empathy is secondary to saving lives through correct diagnosis.",
                round=2,
                phase="REBUTTAL"
            )
        ]
    )
    
    # 2. Initialize Judge
    print("\n[1/2] Initializing Model...")
    try:
        judge = HGTJudge(device="cuda") 
    except Exception as e:
        print(f"❌ CRITICAL: Judge Initialization Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Run Inference
    print("\n[2/2] Running Inference on Mock Debate...")
    try:
        result = judge.judge(dummy_log)
        print("\n✅ INFERENCE SUCCESS!")
        print(f"   Verdict: {result['verdict']}")
        print(f"   Score:   {result['score']:.4f}")
        print(f"   Reason:  {result['reason']}")
    except Exception as e:
        print(f"❌ CRITICAL: Inference Failed: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n🎉 SYSTEM IS STABLE. READY FOR DEMO.")

if __name__ == "__main__":
    test_pipeline()