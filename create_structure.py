#!/usr/bin/env python3
"""
Script to create the training_generator_agent folder structure.
Run this from inside the training_generator_agent directory.
"""
import os
from pathlib import Path


def create_structure():
    """Create the complete folder structure for the training generator agent."""

    # Define the directory structure
    directories = [
        "src/agents/training_generator/nodes",
        "src/agents/training_generator/tools",
        "src/agents/training_generator/prompts",
        "src/agents/training_generator/utils",
    ]

    # Define files to create with their paths and initial content
    files = {
        # Core module files
        "src/__init__.py": "",
        "src/agents/__init__.py": "",
        "src/agents/training_generator/__init__.py": "",
        "src/agents/training_generator/nodes/__init__.py": "",
        "src/agents/training_generator/tools/__init__.py": "",
        "src/agents/training_generator/prompts/__init__.py": "",
        "src/agents/training_generator/utils/__init__.py": "",

        # State schema
        "src/agents/training_generator/state.py": """from typing import TypedDict, List, Optional, Dict, Any


class TrainingGeneratorState(TypedDict):
    \"""State schema for the training generator agent.\"""
    # Input
    input_text: str

    # Processing
    processed_data: Optional[Dict[str, Any]]

    # Output
    generated_training: Optional[str]

    # Metadata
    metadata: Optional[Dict[str, Any]]
""",

        # Pydantic models
        "src/agents/training_generator/models.py": """from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class TrainingInput(BaseModel):
    \"""Input model for training generation.\"""
    text: str = Field(..., description="Input text for training generation")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Additional options")


class TrainingOutput(BaseModel):
    \"""Output model for generated training content.\"""
    content: str = Field(..., description="Generated training content")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Output metadata")
""",

        # Requirements
        "requirements.txt": """# Core dependencies
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0
pydantic>=2.0.0
python-dotenv>=1.0.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
""",

        # Environment template
        ".env.example": """# API Keys
OPENAI_API_KEY=your_openai_api_key_here

# Model Configuration
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.7

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
""",

        # Git ignore
        ".gitignore": """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
*.log
logs/
""",

        # README
        "README.md": """# Training Generator Agent

A LangGraph-based agent for generating training content.

## Project Structure

```
training_generator_agent/
├── src/agents/training_generator/
│   ├── state.py              ✅ State schema with TypedDict
│   ├── models.py             ✅ Pydantic models for structured outputs
│   ├── nodes/                📁 Agent nodes (Ready for Step 2)
│   ├── tools/                📁 Agent tools (Ready for Step 3)
│   ├── prompts/              📁 Prompt templates (Ready for Step 2)
│   └── utils/                📁 Utility functions (Ready for Step 5)
├── requirements.txt          ✅ All dependencies
├── .env.example              ✅ Configuration template
├── .gitignore                ✅ Git ignore rules
├── README.md                 ✅ Complete documentation
└── test_step1.py             ✅ Validation tests
```

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run validation tests:
   ```bash
   python test_step1.py
   ```

## Development

- **Step 1**: ✅ Project structure and state management
- **Step 2**: 📁 Implement agent nodes and prompts
- **Step 3**: 📁 Add tools and integrations
- **Step 4**: 📁 Build the LangGraph workflow
- **Step 5**: 📁 Testing and utilities

## License

MIT
""",

        # Test script
        "test_step1.py": """#!/usr/bin/env python3
\"""Validation tests for Step 1: Project structure and state management.\"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.training_generator.state import TrainingGeneratorState
from src.agents.training_generator.models import TrainingInput, TrainingOutput


def test_state_schema():
    \"""Test state schema creation.\"""
    print("Testing state schema...")

    state: TrainingGeneratorState = {
        "input_text": "Test input",
        "processed_data": None,
        "generated_training": None,
        "metadata": None
    }

    assert state["input_text"] == "Test input"
    print("✅ State schema works correctly")


def test_pydantic_models():
    \"""Test Pydantic models.\"""
    print("\\nTesting Pydantic models...")

    # Test input model
    input_model = TrainingInput(text="Sample input")
    assert input_model.text == "Sample input"
    print("✅ TrainingInput model works")

    # Test output model
    output_model = TrainingOutput(content="Generated content")
    assert output_model.content == "Generated content"
    print("✅ TrainingOutput model works")


def test_directory_structure():
    \"""Test that all required directories exist.\"""
    print("\\nTesting directory structure...")

    required_dirs = [
        "src/agents/training_generator/nodes",
        "src/agents/training_generator/tools",
        "src/agents/training_generator/prompts",
        "src/agents/training_generator/utils",
    ]

    for dir_path in required_dirs:
        path = Path(dir_path)
        assert path.exists() and path.is_dir(), f"Missing directory: {dir_path}"
        print(f"✅ {dir_path}")


def test_required_files():
    \"""Test that all required files exist.\"""
    print("\\nTesting required files...")

    required_files = [
        "src/agents/training_generator/state.py",
        "src/agents/training_generator/models.py",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "README.md",
    ]

    for file_path in required_files:
        path = Path(file_path)
        assert path.exists() and path.is_file(), f"Missing file: {file_path}"
        print(f"✅ {file_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1 VALIDATION TESTS")
    print("=" * 60)

    try:
        test_state_schema()
        test_pydantic_models()
        test_directory_structure()
        test_required_files()

        print("\\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Step 1 Complete!")
        print("=" * 60)
        print("\\nNext steps:")
        print("  - Step 2: Implement agent nodes in nodes/")
        print("  - Step 3: Add tools in tools/")

    except AssertionError as e:
        print(f"\\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\\n❌ ERROR: {e}")
        sys.exit(1)
"""
    }

    print("Creating directory structure...")
    print("=" * 60)

    # Create directories
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

    print("\n" + "=" * 60)
    print("Creating files...")
    print("=" * 60)

    # Create files
    for file_path, content in files.items():
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print(f"✅ Created file: {file_path}")

    print("\n" + "=" * 60)
    print("✅ FOLDER STRUCTURE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Install dependencies: pip install -r requirements.txt")
    print("  2. Configure .env: copy .env.example .env")
    print("  3. Edit .env with your API keys")
    print("  4. Run validation: python test_step1.py")
    print("=" * 60)


if __name__ == "__main__":
    create_structure()
