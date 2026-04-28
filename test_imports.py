#!/usr/bin/env python3
"""Test if imports work correctly."""


    # Test dataclass creation
    r = RetrievalResult(chunk_id="test", text="hello", score=0.9)
    print(f"✓ RetrievalResult creation OK: {r.chunk_id}")
    
    print("\n✅ All imports successful!")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
