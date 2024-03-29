from flask import Blueprint
from controllers.optimization_controller import index, form_view, optimize, report_view

blueprint = Blueprint('blueprint', __name__)

blueprint.route('/', methods=['GET'])(index)
blueprint.route('/optimize_deployment', methods=['GET'])(form_view)
blueprint.route('/optimize', methods=['POST'])(optimize)
blueprint.route('/report', methods=['GET'])(report_view)
