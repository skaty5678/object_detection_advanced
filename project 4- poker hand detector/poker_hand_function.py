def find_poker_hand(hand):
    ranks = []
    suits = []
    possible_ranks = []

    for card in hand:
        suit = card[-1]
        if len(card) == 2:
            rank = card[0]
        else:
            rank = card[0:2]
        if rank == 'A':
            rank = 14
        elif rank == 'K':
            rank = 13
        elif rank == 'Q':
            rank = 12
        elif rank == 'J':
            rank = 11
        ranks.append(int(rank))
        suits.append(suit)

    # print(ranks)
    sorted_ranks = sorted(ranks)
    # print(sorted_ranks)

    if suits.count(suits[0]) == 5:
        if all(rank in ranks for rank in [14, 13, 12, 11, 10]):
            possible_ranks.append(10)
        elif all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
            possible_ranks.append(9)
        else:
            possible_ranks.append(6)

    if all(sorted_ranks[i] == sorted_ranks[i - 1] + 1 for i in range(1, len(sorted_ranks))):
        possible_ranks.append(5)

    hand_unique_values = list(set(sorted_ranks))
    if len(hand_unique_values) == 2:
        for val in hand_unique_values:
            if sorted_ranks.count(val) == 4:
                possible_ranks.append(8)
            if sorted_ranks.count(val) == 3:
                possible_ranks.append(7)

    if len(hand_unique_values) == 3:
        for val in hand_unique_values:
            if sorted_ranks.count(val) == 3:
                possible_ranks.append(4)
            if sorted_ranks.count(val) == 2:
                possible_ranks.append(3)

    if len(hand_unique_values) ==4:
        for val in hand_unique_values:
            if sorted_ranks.count(val) == 2:
                possible_ranks.append(2)

    if not possible_ranks:
        possible_ranks.append(1)

    poker_hand_ranks = {10: 'royal flush', 9: 'straight flush', 8: 'four of a kind', 7: 'full house', 6: 'flush',
                        5: 'straight', 4: 'three of a kind', 3: 'two pair', 2: 'pair', 1: 'high card'}

    output = poker_hand_ranks[max(possible_ranks)]
    print(hand, output)
    return output


if __name__ == '__main__':
    find_poker_hand(["KH", "AH", "QH", "JH", "10H"])  # Royal Flush
    find_poker_hand(["QC", "JC", "10C", "9C", "8C"])  # Straight Flush
    find_poker_hand(["5C", "5S", "5H", "5D", "QH"])  # Four of a Kind
    find_poker_hand(["2H", "2D", "2S", "10H", "10C"])  # Full House
    find_poker_hand(["2D", "KD", "7D", "6D", "5D"])  # Flush
    find_poker_hand(["JC", "10H", "9C", "8C", "7D"])  # Straight
    find_poker_hand(["10H", "10C", "10D", "2D", "5S"])  # Three of a Kind
    find_poker_hand(["KD", "KH", "5C", "5S", "6D"])  # Two Pair
    find_poker_hand(["2D", "2S", "9C", "KD", "10C"])  # Pair
    find_poker_hand(["KD", "5H", "2D", "10C", "JH"])  # High Card
