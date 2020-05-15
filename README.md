# card-games-agents
Reinforcement Learning Agents for Card Games
Python Version: 3.7.3

The `environments` folder holds the environments the agents can interact with.
The agents are Player objects that override the method `def selectCard(self, state, env)`

`CardGameEnvironment` is a generic class that is meant to be sub-classed

`ArbamiyyeEnvironment` represents a trick taking card game.

To run a quick game in the terminal: `python3 arbamiyye_test.pypython3 arbamiyye_test.py`

Rule rundown: 
- There are 13 cards in each of the 4 players' hands
- Hearts is the trump suite
- The player who won the last round plays first, this is the leading card
- Clockwise, players must put down cards that match the suite of the leading card
- If a player does not have the suite of the leading card, they can play any card in their hand
- The player with the highest trump card wins the round, or if no trump card was played, the player with the highest card with the leading card's suite wins the round
- Rounds are played until players have no cards left (ie. 13 rounds)
- The player with the most rounds won wins the game