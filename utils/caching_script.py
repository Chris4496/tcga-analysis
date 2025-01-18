import functools
import os
import hashlib
import pickle


def cache_result(cache_dir='cache', verbose=0):
    """
    Decorator to cache function results using pickle
    
    Parameters:
    -----------
    cache_dir : str, default='cache'
        Directory to store cached results
    verbose : int, default=0
        0: no output
        1: print cache operations
        2: print cache operations and file paths
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache directory if it doesn't exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                if verbose >= 2:
                    print(f"Created cache directory: {cache_dir}")
            
            # Create a unique cache key based on function name, args, and kwargs
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            key_string = '_'.join(key_parts)
            
            # Create hash of the key string
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
            
            # Try to load cached result
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'rb') as f:
                        if verbose >= 1:
                            print("Loading cached result")
                        if verbose >= 2:
                            print(f"Cache file: {cache_file}")
                        return pickle.load(f)
                except Exception as e:
                    if verbose >= 1:
                        print(f"Error loading cache: {e}")
            
            # Calculate result if not cached
            result = func(*args, **kwargs)
            
            # Save result to cache
            try:
                with open(cache_file, 'wb') as f:
                    if verbose >= 1:
                        print("Saving result to cache")
                    if verbose >= 2:
                        print(f"Cache file: {cache_file}")
                    pickle.dump(result, f)
            except Exception as e:
                if verbose >= 1:
                    print(f"Error saving to cache: {e}")
            
            return result
        return wrapper
    return decorator