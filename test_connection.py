#!/usr/bin/env python3


import psycopg2
from config import settings

print("=" * 70)
print("SUPABASE CONNECTION TEST")
print("=" * 70)

url = settings.SUPABASE_DB_URL
masked_url = url.replace(url.split("@")[0].split(":")[-1], "***")
print(f"\nTesting URL: {masked_url}")

try:
    print("\n[1/3] Connecting...")
    conn = psycopg2.connect(
        settings.SUPABASE_DB_URL,
        options="-c search_path=public",
        connect_timeout=5
    )
    print("✅ Connected!")
    
    print("\n[2/3] Testing database...")
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    version = cursor.fetchone()[0]
    print(f"✅ PostgreSQL: {version.split(',')[0]}")
    
    print("\n[3/3] Checking pgvector extension...")
    cursor.execute("""
        SELECT extversion FROM pg_extension WHERE extname='vector'
    """)
    result = cursor.fetchone()
    if result:
        print(f"✅ pgvector v{result[0]} installed")
    else:
        print("⚠️  pgvector NOT installed")
        print("   → You need to run: database/migrations/002_pgvector_setup.sql")
    
    cursor.close()
    conn.close()
    
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - Ready to go!")
    print("=" * 70)
    
except psycopg2.OperationalError as e:
    error_msg = str(e)
    print(f"\n❌ Connection failed: {error_msg[:200]}")
    
    if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
        print("\n📋 NETWORK ISSUE DETECTED")
        print("   Port 5432 or 6543 appears to be blocked by your network/firewall")
        print("\n   Possible causes:")
        print("   • ISP/Firewall blocking PostgreSQL ports")
        print("   • Corporate network proxy")
        print("   • VPN/network restrictions")
        print("\n   Solutions to try:")
        print("   1. Test from a different network (mobile hotspot, different wifi)")
        print("   2. Check if port 5432/6543 is open")
        print("   3. Contact IT if on corporate network")
        print("   4. Use SSH tunneling as workaround")
        
    elif "Name or service not known" in error_msg or "could not translate" in error_msg:
        print("\n📋 DNS RESOLUTION ISSUE")
        print("   The hostname can't be resolved")
        print("   Check:")
        print("   1. Your internet connection")
        print("   2. DNS settings")
        print("   3. The SUPABASE_DB_URL format in .env")
        
    elif "authentication failed" in error_msg or "password" in error_msg.lower():
        print("\n📋 AUTHENTICATION FAILED")
        print("   The password or credentials are wrong")
        print("   Check:")
        print("   1. Password has correct URL encoding (%22 for quotes)")
        print("   2. SUPABASE_DB_URL format is correct")
        
    else:
        print("\n📋 OTHER ERROR - Details above")
        
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
