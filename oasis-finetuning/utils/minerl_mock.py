"""
Mock minerl module to allow VPT imports without installing the full minerl package.

The full minerl package has complex dependencies (old gym, Java, etc.) that are
difficult to install. This mock provides the minimal interface needed for VPT's
Inverse Dynamics Model (IDM) to work.

Usage:
    Import this module before importing VPT:
        from utils.minerl_mock import install_mock
        install_mock()
"""

import sys
from types import ModuleType


# Mock MINERL_ITEM_MAP - not actually used in IDM prediction path
MINERL_ITEM_MAP = {i: f"item_{i}" for i in range(256)}


def install_mock():
    """Install mock minerl modules into sys.modules."""
    
    # Create mock module hierarchy
    minerl = ModuleType("minerl")
    minerl_herobraine = ModuleType("minerl.herobraine")
    minerl_herobraine_hero = ModuleType("minerl.herobraine.hero")
    minerl_herobraine_hero_mc = ModuleType("minerl.herobraine.hero.mc")
    
    # Add the required attribute
    minerl_herobraine_hero_mc.MINERL_ITEM_MAP = MINERL_ITEM_MAP
    
    # Set up module hierarchy
    minerl.herobraine = minerl_herobraine
    minerl_herobraine.hero = minerl_herobraine_hero
    minerl_herobraine_hero.mc = minerl_herobraine_hero_mc
    
    # Install into sys.modules
    sys.modules["minerl"] = minerl
    sys.modules["minerl.herobraine"] = minerl_herobraine
    sys.modules["minerl.herobraine.hero"] = minerl_herobraine_hero
    sys.modules["minerl.herobraine.hero.mc"] = minerl_herobraine_hero_mc
    
    print("  ✓ Installed minerl mock module")

