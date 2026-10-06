---
id: pol_taxes
title: Sales Tax
topic: taxes
store: Northstar Market
---

# Sales Tax

## overview

Sales tax is calculated at checkout from the destination and the item. The assistant does not recompute a rate in chat and does not accept an exemption verbally.

## rules

The tax on the order is the tax that was charged. An exemption has to be on the account through the exemption flow before purchase. Describing a status in chat does not apply one afterward. When an item is refunded, any tax portion follows the refund article and is not given its own quoted amount unless the order shows that amount. Duties on international orders are not sales tax.

## examples

A customer wants the tax refunded without a returned item and without an exemption on the order. Do not promise that refund. Point to the tax line on the order.

A customer asks the assistant to "just recalculate" a rate from the city name. Refuse the calculation. The order total is the record.

## agent_rules

Do not invent a tax rate or a refunded tax amount. Do not ask for a government identifier in chat. Do not treat a duty question as a sales-tax recalculation.


## repeated_reminders

This repeated block is boilerplate that rides along with Sales Tax. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Sales Tax. None of those figures changes the decision section of Sales Tax. If the customer quotes one of those other figures while asking about Sales Tax, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
