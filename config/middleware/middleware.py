import uuid


class UserIdentifierMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        user_identifier = request.COOKIES.get('user_identifier')
        if not user_identifier:
            user_identifier = str(uuid.uuid4())
            request.user_identifier = user_identifier
            response = self.get_response(request)
            response.set_cookie('user_identifier', user_identifier)
            return response
        request.user_identifier = user_identifier
        return self.get_response(request)
