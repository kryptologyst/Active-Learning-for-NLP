#!/usr/bin/env python3
"""
Setup script for Active Learning NLP project.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} is not supported. Please use Python 3.8 or higher.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    directories = ['data', 'models', 'results', 'logs', 'config']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if pip is available
    if not run_command("pip --version", "Checking pip"):
        print("❌ pip is not available. Please install pip first.")
        return False
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def create_default_config():
    """Create default configuration file."""
    print("⚙️ Creating default configuration...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / "src"))
        from config import create_default_config_file
        
        create_default_config_file("config/default.yaml")
        print("✅ Default configuration created")
        return True
    except Exception as e:
        print(f"❌ Failed to create default configuration: {e}")
        return False


def test_installation():
    """Test the installation."""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        sys.path.append(str(Path(__file__).parent / "src"))
        from active_learning import ActiveLearningPipeline
        from data_utils import DataManager
        from config import Config
        
        print("✅ All imports successful")
        
        # Test basic functionality
        config = Config()
        data_manager = DataManager()
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Active Learning NLP Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed during dependency installation")
        sys.exit(1)
    
    # Create default config
    if not create_default_config():
        print("❌ Setup failed during configuration creation")
        sys.exit(1)
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during testing")
        sys.exit(1)
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the web interface: streamlit run web_app/app.py")
    print("2. Run CLI examples: python cli.py --help")
    print("3. Run example script: python example.py")
    print("4. Check the README.md for more information")


if __name__ == "__main__":
    main()
