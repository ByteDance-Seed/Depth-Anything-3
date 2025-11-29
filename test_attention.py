#!/usr/bin/env python3
"""Test script for optimized attention implementation."""

import torch
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import just the attention module without full package dependencies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "optimized_attention",
    "src/depth_anything_3/model/optimized_attention.py"
)
optimized_attention = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimized_attention)

scaled_dot_product_attention_optimized = optimized_attention.scaled_dot_product_attention_optimized
should_use_manual_attention = optimized_attention.should_use_manual_attention

def test_optimized_attention():
    """Test the optimized attention implementation."""
    # Test on available device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f'Testing on device: {device}')
    print(f'Should use manual attention: {should_use_manual_attention(device)}')

    # Create test tensors
    B, H, N, D = 2, 8, 64, 64  # batch, heads, sequence, head_dim
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)
    v = torch.randn(B, H, N, D, device=device)

    # Test without mask
    print('\n1. Testing without attention mask...')
    output = scaled_dot_product_attention_optimized(q, k, v)
    print(f'   Output shape: {output.shape}')
    assert output.shape == (B, H, N, D), f'Expected shape {(B, H, N, D)}, got {output.shape}'
    print('   ✅ Passed')

    # Test with mask (should be expanded to include heads dimension)
    print('\n2. Testing with attention mask...')
    attn_mask = torch.zeros(B, H, N, N, device=device)  # B, H, N, N format
    output_masked = scaled_dot_product_attention_optimized(q, k, v, attn_mask=attn_mask)
    print(f'   Output shape: {output_masked.shape}')
    assert output_masked.shape == (B, H, N, D), f'Expected shape {(B, H, N, D)}, got {output_masked.shape}'
    print('   ✅ Passed')

    # Test with dropout (training mode)
    print('\n3. Testing with dropout (training mode)...')
    output_dropout = scaled_dot_product_attention_optimized(
        q, k, v, dropout_p=0.1, training=True
    )
    print(f'   Output shape: {output_dropout.shape}')
    assert output_dropout.shape == (B, H, N, D), f'Expected shape {(B, H, N, D)}, got {output_dropout.shape}'
    print('   ✅ Passed')

    # Test with custom scale
    print('\n4. Testing with custom scale...')
    output_scaled = scaled_dot_product_attention_optimized(q, k, v, scale=0.125)
    print(f'   Output shape: {output_scaled.shape}')
    assert output_scaled.shape == (B, H, N, D), f'Expected shape {(B, H, N, D)}, got {output_scaled.shape}'
    print('   ✅ Passed')

    # Verify numerical correctness on CPU (compare with PyTorch implementation)
    if device.type != 'cpu':
        print('\n5. Testing numerical correctness (CPU comparison)...')
        q_cpu = q.cpu()
        k_cpu = k.cpu()
        v_cpu = v.cpu()

        output_optimized = scaled_dot_product_attention_optimized(q_cpu, k_cpu, v_cpu).numpy()
        output_pytorch = torch.nn.functional.scaled_dot_product_attention(q_cpu, k_cpu, v_cpu).numpy()

        # Check if outputs are close
        max_diff = abs(output_optimized - output_pytorch).max()
        print(f'   Max difference from PyTorch implementation: {max_diff:.6f}')
        assert max_diff < 1e-5, f'Outputs differ too much: {max_diff}'
        print('   ✅ Passed')

    print('\n' + '='*60)
    print('✅ All tests passed!')
    print('='*60)

if __name__ == '__main__':
    test_optimized_attention()
