---
id: pol_payments
title: Payment Methods and Failed Payments
topic: payments
store: Northstar Market
---

# Payment Methods and Failed Payments

## overview

Payment methods are chosen at checkout and can change by region, item, and cart. This article covers declines, pending authorizations, and captured charges. It also lists the payment data the assistant must never collect.

## rules

If a payment fails, the customer should verify the details shown at checkout and check with the bank or provider for a decline or an authentication request. If the bank shows a pending authorization, the customer should not keep submitting payment. They should check the order status and the payment activity first. A pending authorization is not a completed charge. It can expire without becoming a capture. A captured charge is an actual payment and is tied to the order.

## exceptions

Retrying checkout does not reverse a capture. Two statement lines, one pending and one captured, are not automatically two charges. The pending line can release while the captured line remains the order payment. A refund of a captured payment follows the refund article and needs the order record. Gift cards are entered only in the official field. This article does not reveal or accept a card number.

## steps

Identify the status the customer can actually see: declined, pending authorization, or captured. For a decline with no order, point them to the bank or provider, then a new checkout attempt. For a pending authorization, tell them to stop resubmitting until they have checked whether an order exists. For a capture, use the order. Never request a full card number, CVV, password, or one-time passcode.

## examples

Checkout says the card was declined and no order exists. Suggest checking the bank. Do not take the card number in chat.

The bank shows a pending authorization and the customer wants to press pay again. Tell them to stop and check the order status. Pending is not captured.

The customer says that pending authorization is a completed charge. It is not.

The statement shows one pending authorization and one captured charge. Explain that the pending amount may drop off and that only the capture is the order payment. Do not refund both from chat.

The customer offers a CVV and a one-time passcode so the agent can "finish" the decline. Refuse the data and keep the customer on the bank check or the official checkout.

## agent_rules

Never ask for a full card number, CVV, password, or one-time passcode. Never advise repeated submission while an authorization is pending. Never call a pending authorization a completed charge.


## repeated_reminders

This repeated block is boilerplate that rides along with Payment Methods and Failed Payments. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Payment Methods and Failed Payments. None of those figures changes the decision section of Payment Methods and Failed Payments. If the customer quotes one of those other figures while asking about Payment Methods and Failed Payments, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
