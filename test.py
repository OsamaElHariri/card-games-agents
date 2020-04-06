from environments.CardGameEnvironment import CardGameEnvironment
from environments.ArbamiyyeEnvironment import ArbamiyyeEnvironment
from players.Player import Player
import cards.CardUtils as CardUtils
import asyncio


players = [Player(), Player(), Player(), Player()]
env = CardGameEnvironment(players, 13)
for player in env.players:
    print(player.cards)

card = env.players[0].cards[0]
env.players[0].play(card)
print("First player after playing {}, {}".format(card, env.players[0].cards))

print("Resetting")
env.reset()
for player in env.players:
    print(player.cards)

print("Utils")
suite, rank = CardUtils.indexToSuiteAndRank(36)
print(suite)
print(rank)

index = CardUtils.suiteRankToIndex(suite, rank)
print(index)

indexTest = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(indexTest[-4:])

# def mark_done(future, result):
#     print('setting future result to {!r}'.format(result))
#     future.set_result(result)


# async def main():
#     loop = asyncio.get_event_loop()
#     future = loop.create_future()
#     print('scheduling mark_done')
#     loop.call_soon(mark_done, future, 'the result')
#     print('suspending the coroutine')
#     result = await future
#     print('awaited result: {!r}'.format(result))
#     print('future result: {!r}'.format(future.result()))
#     return result

# async def tester():
#     print('entering the event loop')
#     result = await main()
#     print('returned result: {!r}'.format(result))

# asyncio.run(tester())

async def genTest(x):
    print("Called Count = {}".format(x))
    while True:
        print("While Start Called Count = {}".format(x))
        x = yield x * 2
        print("While End Called Count = {}".format(x))

async def asyncGenTest():
    gen = genTest(2)
    print("Line 64")
    print(await gen.asend(None))
    print("Line 66")
    print(await gen.asend(10))
    print("Line 68")
    return 1

asyncGenTestResponse = asyncio.run(asyncGenTest())
print(asyncGenTestResponse)