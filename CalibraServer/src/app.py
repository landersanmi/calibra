from flask import Flask
from routes.blueprint import blueprint


def create_app():
    app = Flask(__name__)
    return app


app = create_app()
app.register_blueprint(blueprint)


if __name__ == '__main__':
    app.run(port=7070)
