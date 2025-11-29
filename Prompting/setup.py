"""Setup script for easy installation and testing."""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and print status."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed")
        print(e.stderr)
        return False


def main():
    """Main setup script."""
    print("\n" + "="*60)
    print("TRANSLATION ORCHESTRATOR - SETUP")
    print("="*60)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("\n✗ Python 3.10+ required")
        print(f"  Current version: {sys.version}")
        sys.exit(1)
    
    print(f"\n✓ Python version: {sys.version.split()[0]}")
    
    # Install dependencies
    if not run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing dependencies"
    ):
        print("\nSetup failed at dependency installation")
        sys.exit(1)
    
    # Create .env if doesn't exist
    if not os.path.exists(".env"):
        print("\n" + "="*60)
        print("Creating .env file from template")
        print("="*60)
        
        try:
            with open(".env.example", "r") as f:
                content = f.read()
            with open(".env", "w") as f:
                f.write(content)
            print("✓ Created .env file")
            print("  Remember to add your API keys!")
        except Exception as e:
            print(f"✗ Failed to create .env: {e}")
    else:
        print("\n✓ .env file already exists")
    
    # Run tests
    print("\n" + "="*60)
    print("Would you like to run tests? (y/n)")
    print("="*60)
    
    response = input("> ").strip().lower()
    if response == 'y':
        run_command(
            f"{sys.executable} -m pytest tests/ -v",
            "Running tests"
        )
    
    # Success message
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Edit .env and add your API keys")
    print("2. Run example: python examples/example_usage.py")
    print("3. Run tests: pytest tests/ -v")
    print("\nFor more information, see README.md")
    print("="*60)


if __name__ == "__main__":
    main()
