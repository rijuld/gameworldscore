#!/usr/bin/env python3
"""
Memory monitoring utility for Oasis GRPO training.
Run this in a separate terminal to monitor GPU memory usage in real-time.
"""

import torch
import time
import sys
from datetime import datetime

def format_bytes(bytes_val):
    """Format bytes to human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def print_memory_stats():
    """Print detailed GPU memory statistics."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    
    device = torch.cuda.current_device()
    
    # Get memory stats
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    max_allocated = torch.cuda.max_memory_allocated(device)
    max_reserved = torch.cuda.max_memory_reserved(device)
    
    # Get total GPU memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    
    # Calculate percentages
    allocated_pct = (allocated / total_memory) * 100
    reserved_pct = (reserved / total_memory) * 100
    free_memory = total_memory - reserved
    free_pct = (free_memory / total_memory) * 100
    
    # Print stats
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] GPU Memory Statistics (Device {device}):")
    print(f"{'='*70}")
    print(f"Total Memory:     {format_bytes(total_memory)}")
    print(f"Allocated:        {format_bytes(allocated):>12} ({allocated_pct:>5.1f}%)")
    print(f"Reserved:         {format_bytes(reserved):>12} ({reserved_pct:>5.1f}%)")
    print(f"Free:             {format_bytes(free_memory):>12} ({free_pct:>5.1f}%)")
    print(f"{'='*70}")
    print(f"Peak Allocated:   {format_bytes(max_allocated)}")
    print(f"Peak Reserved:    {format_bytes(max_reserved)}")
    print(f"{'='*70}")
    
    # Memory fragmentation indicator
    fragmentation = reserved - allocated
    if fragmentation > 1e9:  # More than 1GB fragmentation
        print(f"⚠️  High fragmentation: {format_bytes(fragmentation)} reserved but not allocated")
    
    # Warning if memory is critically low
    if free_pct < 5:
        print(f"🚨 CRITICAL: Only {free_pct:.1f}% memory free!")
    elif free_pct < 15:
        print(f"⚠️  WARNING: Only {free_pct:.1f}% memory free")

def monitor_continuous(interval=2):
    """Continuously monitor memory at specified interval (seconds)."""
    print("Starting continuous GPU memory monitoring...")
    print(f"Update interval: {interval} seconds")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            print_memory_stats()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        print_memory_stats()  # Print final stats

def reset_peak_stats():
    """Reset peak memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print("✓ Peak memory statistics reset")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            print_memory_stats()
        elif sys.argv[1] == "--reset":
            reset_peak_stats()
            print_memory_stats()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python monitor_memory.py              # Continuous monitoring (2s interval)")
            print("  python monitor_memory.py --once        # Print stats once and exit")
            print("  python monitor_memory.py --reset       # Reset peak stats and print")
            print("  python monitor_memory.py --interval N  # Monitor with N second interval")
        elif sys.argv[1] == "--interval" and len(sys.argv) > 2:
            interval = float(sys.argv[2])
            monitor_continuous(interval)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        monitor_continuous()
