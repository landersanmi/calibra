from flask import Flask
from routes.blueprint import blueprint

import secrets


def create_app():
    app = Flask(__name__)
    app.secret_key = secrets.token_hex(16)
    return app


if __name__ == '__main__':
    app = create_app()
    app.register_blueprint(blueprint)
    app.run(host="0.0.0.0", port=7070)
