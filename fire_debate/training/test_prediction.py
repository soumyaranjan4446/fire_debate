import sys
import os
import torch

# --- Path Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from fire_debate.insight.hgt_judge import HGTJudge
from fire_debate.schemas.debate import DebateLog, DebateTurn

def test_manual_prediction():
    # 1. Path to your saved model
    model_path = os.path.join(project_root, "data", "processed", "hgt_judge.pth")
    
    # 2. Initialize the Judge
    try:
        judge = HGTJudge(model_path=model_path, device="cuda")
    except Exception as e:
        print(f"❌ Failed to load judge: {e}")
        return

    print("\n📝 Creating a Mock Debate for Testing...")

    # 3. Create a Dummy Debate (Topic: Nuclear Energy)
    mock_log = DebateLog(
        debate_id="test_001",
        claim_id="claim_001",
        claim_text="Nuclear energy is essential for reaching net-zero carbon emissions.",
        ground_truth=True, 
        turns=[
            DebateTurn(
                turn_id="t1",
                text="Nuclear power provides a consistent baseload of carbon-free electricity that solar and wind cannot match without massive battery storage.",
                citations=["IEA Report 2022"]  # Passed as a list to be safe
            ),
            DebateTurn(
                turn_id="t2",
                text="However, the waste problem remains unsolved and poses long-term environmental risks.",
                citations=["Greenpeace"]
            ),
            DebateTurn(
                turn_id="t3",
                text="Modern reactors produce significantly less waste, and deep geological repositories offer a safe long-term solution. The climate crisis is a more immediate threat.",
                citations=["World Nuclear Association"]
            )
        ]
    )

    # 4. Run Prediction
    print(f"🔎 Analyzing Claim: '{mock_log.claim_text}'")
    
    try:
        # GraphBuilder will process the text and the model will infer relevance
        result = judge.judge(mock_log)

        # 5. Print Results
        print("\n" + "="*30)
        print("🤖 MODEL PREDICTION")
        print("="*30)
        print(f"Verdict:    {'✅ TRUE' if result['verdict'] else '❌ FALSE'}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Raw Score:  {result['score']:.4f}")
        print(f"Reason:     {result['reason']}")
        print("="*30)
    except Exception as e:
        print(f"❌ Prediction Failed: {e}")
        print("💡 Hint: Check if 'citations' field name matches your Schema exactly.")

if __name__ == "__main__":
    test_manual_prediction()