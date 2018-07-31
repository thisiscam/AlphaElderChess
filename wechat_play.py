from wxpy import *
from PIL import Image, ImageDraw

from elder_chess_native import Move

from xmlrpc.client import ServerProxy
chess_api = ServerProxy("http://127.0.0.1:1027")

bot = Bot(cache_path=True, console_qr=True)

cam = bot.friends().search("Cam")[0]

cam.send("Connected!")

listen_to_usrs = ["超级赛亚源", "李京津", "额勒金德", "张小三", "唐鹏", "Cam", "null", "绅士是共通的", "王同学"]
listen_to_chats = []
for user in listen_to_usrs:
	users = bot.friends().search(user)
	groups = bot.groups().search(user)
	if users:
		listen_to_chats += list(users)
	if groups:
		listen_to_chats += list(groups)

def send_board(msg, begin_txt, end_txt):
	msg.reply(begin_txt)
	board_txt = chess_api.display_board()
	image = Image.new("RGBA", (110,125), (255,255,255))
	draw = ImageDraw.Draw(image)
	draw.text((5, 5), board_txt, (0,0,0))
	image.save("tmp.png")
	msg.reply_image("tmp.png")
	msg.reply(end_txt)

def winning_message(msg):
	winner = chess_api.get_winner()
	if winner == 0:
		rmsg = "W赢了"
	elif winner == 1:
		rmsg = "B赢了"
	elif winner == 2:
		rmsg = "和局了"
	send_board(msg, "游戏结束!", rmsg)

@bot.register(chats=listen_to_chats, msg_types=[TEXT], except_self=False)
def reply_loop(msg):
	print(msg.text)
	if msg.text.startswith("开始游戏"):
		print("game start api")
		try:
			chess_api.start_game()
		except Exception as e:
			print(e)
		send_board(msg, "游戏开始; 回复 '怎么玩' 获取走棋方法", "请走第一步")
	elif msg.text.startswith("怎么玩"):
		msg.reply(
			"""游戏方法: 
				回复 'f 1 0' 表示 '翻开 (1,0)'
				回复 'm 1 0 u' 表示 '移动 (1,0) 向上'
			""")
	else:
		move_str = msg.text
		if not chess_api.make_move(move_str):
			return "Invalid Move"
		if chess_api.game_ended():
			return winning_message(msg)
		else:
			send_board(msg, "你走了 {0}".format(move_str), "请等待AI的回应")
			ai_move = chess_api.ai_make_move()
			if chess_api.game_ended():
				return winning_message(msg)
			rmsg = str(ai_move)
			send_board(msg, "AI走 {0}".format(rmsg), "请走下一步")

embed()
