# file: backend/fix_ssl.py
import os
import ssl
import certifi

def fix_ssl_certificates():
    """
    Finds the path to the certifi certificate bundle and configures Python's
    SSL context to use it, resolving the CERTIFICATE_VERIFY_FAILED error on macOS.
    """
    cert_path = certifi.where()
    print(f"Found certifi's certificate bundle at: {cert_path}")

    try:
        # For urllib
        os.environ["SSL_CERT_FILE"] = cert_path
        # For requests
        os.environ["REQUESTS_CA_BUNDLE"] = cert_path

        # Monkey-patch the default SSL context
        ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=cert_path)

        print("✅ SSL certificate context has been successfully updated.")
        print("You can now try running 'python3 app.py' again.")

    except Exception as e:
        print(f"❌ Failed to update SSL context: {e}")

if __name__ == "__main__":
    fix_ssl_certificates()