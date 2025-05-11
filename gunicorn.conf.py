import os

# Bind to the port specified by Render
bind = f"0.0.0.0:{os.getenv('PORT', '8000')}"

# Workers based on environment variable or default to 2
workers = int(os.getenv('GUNICORN_WORKERS', 2))

# Timeout for slow operations
timeout = 120

# Access logging
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log to stderr
loglevel = "info"
