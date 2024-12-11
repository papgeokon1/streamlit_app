import psutil

def check_memory_usage():
    """
    Ελέγχει τη μνήμη RAM που χρησιμοποιείται από τη διαδικασία.
    Επιστρέφει τη χρησιμοποιούμενη RAM σε MB.
    """
    process = psutil.Process()
    memory_info = process.memory_info()
    return f"Χρησιμοποιούμενη RAM: {memory_info.rss / (1024 ** 2):.2f} MB"
