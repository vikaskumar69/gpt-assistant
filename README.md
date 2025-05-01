# gpt-assistant

This repo is for proof of concept of how to build a small scale gpt assistant over a private knowledge base.

Steps to use and test:
1. Install dependencies as required with "pip install <dependency_name (eg: langchain)>". Can't name all, install as required. 
2. Change file path in ingest.py where we are loading the pdf file
3. Run ingest.py
4. Update query as required in qa_engine.py
5. Run qa_engine.py

Additional notes:
1. Currently .env and main.py files are of no functional use