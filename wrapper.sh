# if venv doesn't exist, create it
if [ ! -d "venv" ]; then
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi


sudo venv/bin/python wrapper.py "$@"
