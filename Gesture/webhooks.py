import requests

webhook_url = 'https://discord.com/api/webhooks/1491348841256063018/OPvTA5r8K_awQl-6mqMOF6vBFtHn6J_lt-iqJ_OjvjkwF9E2yddi-A7dnQjue20ABUC9' #GOONS general channel
#webhook_url = 'https://discordapp.com/api/webhooks/1491375623313297429/-EdAezO7Xq_0ea-nAYI1Uf8TiQqbKxmuc-yU-bekjbFPtvZP6zO6HVEra32iClhGo7x0' #GOONS bot channel
#webhook_url = 'https://discord.com/api/webhooks/1491044156049457204/ydvJsp1NeID7bbWk-VWOq9XW4ZENzoggkmprtiwaLI1M7PJbvEDvYYcAe5AazQViVHOS' #Own SERVER
#webhook_url = 'https://discord.com/api/webhooks/1491371971643183154/vtl3xGV-ohnTz_QroWVpi5XRks27fi6DproqIWpLU9Ohlg7M1hY-gnkuZ_m25TdIPDbC' #CoK

def send_webhook(content: str, username: str = "Mike", avatar_url: str = "https://media1.tenor.com/m/_ha2H2_hlhEAAAAC/wazowski-mike.gif") -> None:
    try:
        response = requests.post(webhook_url, json={"content": content, "username": username, "avatar_url": avatar_url})
        response.raise_for_status()  # Raise an exception for unsuccessful status codes
        print("Webhook sent successfully")
    except requests.exceptions.RequestException as e:
        print("Error sending webhook:", e)

def send_webhook_middle_finger() -> None:
    #send_webhook("🖕")
    send_webhook("https://tenor.com/view/attackontitan-gif-23614886")

def send_webhook_time_out() -> None:
    #send_webhook("TIMEOUT\n" + "\u200b\n" * 100 + ".")
    send_webhook("https://giphy.com/gifs/news-shaq-timeout-shaw-ltoVrEYgv30GJvSDOk")

def send_webhook_love_heart() -> None:
    send_webhook("💖\n"*20)

def send_webhook_tate() -> None:
    send_webhook("https://tenor.com/view/andrew-tate-bugatti-gif-27497986")

def send_webhook_jjk() -> None:
    send_webhook("⛩️\n"*20)

def send_webhook_gang_signs() -> None:
    send_webhook("NIGGAS IN THE BUILDING 🧑🏿‍🦱")

def send_webhook_kawaii() -> None:
    send_webhook("https://giphy.com/gifs/j-juniel-choi-junhee-y2a2Jm6KNb2PC")

def send_webhook_angry() -> None:
    send_webhook("https://tenor.com/view/nanami-aoyama-%D0%BD%D0%B0%D0%BD%D0%B0%D0%BC%D0%B8-%D0%B0%D0%BE%D1%8F%D0%BC%D0%B0-%D0%BD%D0%B0%D0%BD%D0%B0%D0%BC%D0%B8-%D0%B0%D0%BE%D1%8F%D0%BC%D0%B0-nanami-gif-646161818462015465")

def send_webhook_sob() -> None:
    send_webhook("https://tenor.com/view/scared-lbp-sonic-lbp-little-big-planet-sackboy-gif-27160964")