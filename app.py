import streamlit.web.cli as stcli
import sys

def main():
    sys.argv = ["streamlit", "run", "frontend/dashboard.py"]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
