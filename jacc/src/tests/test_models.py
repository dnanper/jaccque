"""Test script for model providers.

Run with: python test_models.py

Tests:
1. API Model (Gemini) - requires GEMINI_API_KEY
2. HF Inference Model - requires HF_TOKEN
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.model import get_model, APIModel, HFInferenceModel


def test_api_model_gemini():
    """Test API model with Gemini."""
    print("\n" + "="*50)
    print("Testing API Model (Gemini)")
    print("="*50)
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set, skipping...")
        return False
    
    try:
        # Using get_model factory
        model = get_model("api", model_name="gemini/gemini-2.0-flash", max_tokens=100)
        
        response = model.query([
            {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
        ])
        
        print(f"✅ Response: {response['content'][:100]}...")
        print(f"   Calls: {model.n_calls}, Cost: ${model.cost:.6f}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_api_model_convenience():
    """Test API model convenience methods."""
    print("\n" + "="*50)
    print("Testing API Model Convenience Methods")
    print("="*50)
    
    if not os.getenv("GEMINI_API_KEY"):
        print("⚠️  GEMINI_API_KEY not set, skipping...")
        return False
    
    try:
        # Using convenience class method
        model = APIModel.gemini(model="gemini-2.0-flash", max_tokens=50)
        
        response = model.query([
            {"role": "user", "content": "Reply with just 'OK'"}
        ])
        
        print(f"✅ Response: {response['content']}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_hf_inference_model():
    """Test HuggingFace Inference API model."""
    print("\n" + "="*50)
    print("Testing HF Inference Model")
    print("="*50)
    
    if not os.getenv("HF_TOKEN"):
        print("⚠️  HF_TOKEN not set, skipping...")
        return False
    
    try:
        model = get_model(
            "hf_inference",
            model_name="meta-llama/Llama-3.2-1B-Instruct",
            max_tokens=50
        )
        
        response = model.query([
            {"role": "user", "content": "Say hello in one sentence."}
        ])
        
        print(f"✅ Response: {response['content'][:100]}...")
        print(f"   Calls: {model.n_calls}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("Model Providers Test Suite")
    print("="*50)
    
    results = []
    
    # Test API model
    results.append(("API Model (Gemini)", test_api_model_gemini()))
    results.append(("API Model Convenience", test_api_model_convenience()))
    
    # Test HF Inference
    results.append(("HF Inference Model", test_hf_inference_model()))
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "⚠️  SKIP/FAIL"
        print(f"  {status}: {name}")
    
    print(f"\nTotal: {passed}/{total} passed")


if __name__ == "__main__":
    main()
