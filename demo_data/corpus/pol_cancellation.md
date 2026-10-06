---
id: pol_cancellation
title: Order Cancellation
topic: cancellation
store: Northstar Market
---

# Order Cancellation

## overview

Cancellation stops an order or a line item before fulfillment finishes. It is not a return and it is not a refund. The assistant can explain the control and the status. It cannot change the order from chat. A sentence such as "please cancel" is not a completed cancellation.

## rules

An order or item can be cancelled only while the order page shows Cancel item or Cancel order. The customer must use that control. Once packing has started or the carrier has the shipment, the control usually disappears and cancellation is no longer available. The status must read Cancelled before anyone may say the order was cancelled. Until then it is not cancelled, even if the customer asked earlier.

## exceptions

If cancellation is unavailable, the later path is a return after delivery, and only when the item is eligible under the returns article. A preorder whose ship date moves does not cancel itself. In a multi-item order, one line can be cancellable while another has already shipped. Only the line that still shows the control can be cancelled. Membership cancellation is a different flow under Account, then Memberships. Marketplace orders may not show the Northstar control.

## steps

Open the order in the supplied context. If the cancel control is present, tell the customer to use it and to wait until the status says Cancelled. If the control is absent because the carrier has the parcel, say cancellation looks unavailable and mention a later eligibility check under the returns article. Do not state a refund amount. Do not collect payment details.

## examples

Order NS-22011 is Processing and Cancel order is visible. Show the customer that path. Do not say the assistant cancelled it.

Order NS-22018 shipped this morning and no cancel control is present. A chat request does not cancel it. Say the handoff usually closes cancellation.

The customer says the order was cancelled yesterday, but the status is still Processing. Correct that. Conversation is not a status.

Two lines, one shipped and one still Processing with a cancel control. The shipped line stays in transit. The customer can still cancel the processing line on the page.

## agent_rules

Do not claim an order is cancelled unless the supplied status is Cancelled. Do not promise refund timing at the moment of cancellation. Do not confuse this path with cancelling Northstar Plus.


## repeated_reminders

This repeated block is boilerplate that rides along with Order Cancellation. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Order Cancellation. None of those figures changes the decision section of Order Cancellation. If the customer quotes one of those other figures while asking about Order Cancellation, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
