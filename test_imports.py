import time
print('Starting import checks...')

modules = [
    'dotenv',
    'langchain_community',
    'langchain_core',
    'langchain_openai',
    'langgraph',
    'openai',
]

for m in modules:
    print(f'Importing {m}...')
    start = time.time()
    try:
        __import__(m)
        print(f'  OK ({time.time()-start:.2f}s)')
    except Exception as e:
        print(f'  FAILED: {e}')

print('Import checks completed.')
