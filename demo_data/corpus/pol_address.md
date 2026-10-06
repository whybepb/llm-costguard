---
id: pol_address
title: Order Changes and Address Updates
topic: address
store: Northstar Market
---

# Order Changes and Address Updates

## overview

Some orders can still change an address, a delivery option, or a line item. The order page shows whether the edit exists. The assistant cannot type a new address into the account, and a chat message does not save one.

## rules

The customer may edit an address, delivery option, or item only while that option is available in the order details. They should make the change there. Once the order enters fulfillment or a carrier has it, the edit is usually gone. If the address cannot be changed, the customer should use the carrier or support options shown on that order. The assistant must not say an address was saved unless the order result in context says so.

## exceptions

For privacy, do not repeat a full delivery address unless that repeat is necessary and the address is already in the authorized context. Even then, repeat only the part needed. A gift recipient's address gets the same treatment. Cancellation is a different control. If the address edit is still available, an address mistake does not require cancellation. A shipped order is not updated by restating the new street in chat.

## steps

Check the supplied order for an edit control. If it is available, direct the customer to use it and to wait until the page shows the new details. If the carrier already has the parcel, say the edit looks unavailable. Do not confirm a change that the order system has not confirmed. Do not read a full address back as a courtesy.

## examples

The order still shows Edit address. Tell the customer to use that control. Do not claim the new address is already saved.

The carrier picked the parcel up an hour ago and no edit control is present. The change is likely unavailable. Point to the options on the order.

The customer says the address was already changed to a specific street. The context has no confirmation of that save. Say it cannot be confirmed. Do not repeat the full street address.

The customer wants to cancel only because the address is wrong, and Edit address is still present. Explain the edit path first. Do not cancel in chat.

## agent_rules

Never promise that an address, delivery option, or item change was saved unless the order context confirms it. Do not recite a full delivery address back. Do not collect a card number to verify an address change.


## repeated_reminders

This repeated block is boilerplate that rides along with Order Changes and Address Updates. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Order Changes and Address Updates. None of those figures changes the decision section of Order Changes and Address Updates. If the customer quotes one of those other figures while asking about Order Changes and Address Updates, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
