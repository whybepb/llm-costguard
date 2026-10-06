---
id: pol_pickup
title: Store Pickup
topic: pickup
store: Northstar Market
---

# Store Pickup

## overview

Store pickup sends an item to a counter rather than a home address. Pickup status does not rewrite return eligibility. The assistant does not invent a pickup code or a hold length.

## rules

The order page shows when the item is ready and which store is holding it. The customer brings the confirmation and the identification the store requests. Quote a pickup code only when that code is in the order context. The hold time is the time printed on the order, not a guessed number of days. After the customer picks the item up, a return follows the returns article, counted from the pickup time shown on the order.

## examples

The customer missed a pickup and wants an immediate refund. Read the hold status on the order before discussing a refund. Do not invent a 3-day hold. Do not read out an identification number.

The customer asks for the pickup code and the context has none. Say it is not in the provided context. Do not create a code.

## agent_rules

Do not invent a pickup code, a hold period, or a ready status. Do not treat a pickup hold as the 30 calendar day return window.


## repeated_reminders

This repeated block is boilerplate that rides along with Store Pickup. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Store Pickup. None of those figures changes the decision section of Store Pickup. If the customer quotes one of those other figures while asking about Store Pickup, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
