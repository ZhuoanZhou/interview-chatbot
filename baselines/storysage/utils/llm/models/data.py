class ModelResponse:
    def __init__(self, content):
        self.content = content
        self.response_metadata = {}
        
    def get_token_usage(self):
        """
        Extract token usage information from response metadata.

        Returns:
            dict: Token usage with keys 'prompt_tokens', 'completion_tokens', 'total_tokens'
                  Returns zeros if no usage information is available.
        """
        if 'token_usage' in self.response_metadata:
            usage = self.response_metadata['token_usage']
            return {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
        return {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0
        }