---
id: pol_subscriptions
title: Subscriptions and Memberships
topic: subscriptions
store: Northstar Market
---

# Subscriptions and Memberships

## overview

Northstar Plus and other memberships are managed in account settings. The renewal date, fee, and benefits shown there override a general chat impression. The assistant cannot click cancel and cannot confirm that a cancellation or a fee refund has been completed.

## rules

Customers review or manage a membership from Account, then Memberships. Cancellation takes effect on the terms shown in that flow. Do not assume the membership ends the moment the customer asks. Do not assume a refund of the fee. A fee refund depends on the plan terms and on whether a benefit was used.

There is one narrow exception. A first-time monthly membership fee can be refunded through the membership page only when all three are true: the fee posted less than 48 hours ago, no membership benefit has been used, and the membership page shows the refund control. Outside that case, do not promise a fee refund. Even inside that case, the assistant has not processed the refund. The customer must use the control, and the page must show the result.

## exceptions

Cancelling a physical order is not cancelling Plus. A membership refund does not cancel a merchandise order. A merchandise order follows the cancellation article. A renewal that already posted is not undone by a sentence in chat. Used benefits, including free shipping during the term, remove the narrow 48-hour exception. An annual fee is outside that monthly exception. Marketplace purchases are not memberships.

## steps

Direct the customer to Account, then Memberships. If they want an end date, tell them the on-screen terms control it. If they want a fee refund, check the three conditions before describing the narrow exception. If any condition fails, say the refund is not promised. Never ask for a password or a card number.

## examples

The customer asks where to cancel Plus and is not disputing a charge. Give the Account, then Memberships path. Do not say it is already cancelled.

The fee posted 2 hours ago, no benefit was used, and the refund control is visible. This is the narrow case. Tell them to use the control. Do not say the refund has been issued.

The fee posted 10 days ago. That is outside 48 hours, so the narrow full-fee exception does not apply. Do not promise a refund.

The customer says cancelling ends Plus immediately and reverses a renewal that already posted. Do not agree. Immediate termination and an automatic reversal are not the general rule.

The customer says "cancel my order" and the record is a physical order still processing, not a membership. Use the cancellation article.

## agent_rules

Do not confirm a membership cancellation without a system result. Do not promise an immediate end. Do not promise a full fee refund outside the narrow 48-hour case, and do not claim the refund was issued inside that case either. Do not ask for the account password.


## repeated_reminders

This repeated block is boilerplate that rides along with Subscriptions and Memberships. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Subscriptions and Memberships. None of those figures changes the decision section of Subscriptions and Memberships. If the customer quotes one of those other figures while asking about Subscriptions and Memberships, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
