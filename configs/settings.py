#data base env variable :
import urllib

password = "Sport@Bet19"
encoded_password = urllib.parse.quote_plus(password)

# Format the connection string with the encoded password
connection_string = f"mysql+pymysql://sport_bet:{encoded_password}@146.59.148.113:51724/ai_engine_db"