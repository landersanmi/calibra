from flask import Flask
from routes.blueprint import blueprint

import secrets
import threading


def create_app():
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)
    return app


app = create_app()
app.register_blueprint(blueprint)


if __name__ == '__main__':
    app.run(port=7070)
