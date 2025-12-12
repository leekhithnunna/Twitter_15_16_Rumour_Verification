#!/usr/bin/env python3
"""
Setup script for Twitter Rumor Detection Project
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError:
        print("❌ Failed to install requirements. Please install manually:")
        print("pip install -r requirements.txt")
        return False
    return True

def check_data_files():
    """Check if data files exist"""
    required_files = [
        "twitter15/source_tweets.txt",
        "twitter15/label.txt", 
        "twitter16/source_tweets.txt",
        "twitter16/label.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all Twitter15 and Twitter16 data files are present.")
        return False
    
    print("✅ All data files found!")
    return True

def create_directories():
    """Create necessary directories"""
    dirs = ["Results", "Best_of_three"]
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    print("✅ Directories created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Twitter Rumor Detection Project...")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        return
    
    # Create directories
    create_directories()
    
    # Check data files
    if not check_data_files():
        print("\n📝 Next steps:")
        print("1. Add your Twitter15 and Twitter16 data files")
        print("2. Run: python combine_twitter_data.py")
        print("3. Run: python add_numeric_labels.py") 
        print("4. Run: python select_best_three_models.py")
        return
    
    print("\n🎉 Setup complete! You can now run:")
    print("1. python combine_twitter_data.py")
    print("2. python add_numeric_labels.py")
    print("3. python select_best_three_models.py")

if __name__ == "__main__":
    main()