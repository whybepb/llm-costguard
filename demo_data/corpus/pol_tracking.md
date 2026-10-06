---
id: pol_tracking
title: Order Tracking
topic: tracking
store: Northstar Market
---

# Order Tracking

## overview

Tracking reports carrier scans. This assistant has no live carrier login. It may quote a tracking result only when that result is included in the request context. It may not extend that result with a guessed city, depot, or date.

## rules

The latest status is in Orders, on the purchase. After a parcel is handed to the carrier, a new scan can take up to 24 hours to appear. No scan during those 24 hours is not evidence that the parcel is lost. A delivery estimate is not a guarantee. If the context includes a tracking number and scans, those are the only carrier facts to use. If it includes neither, do not create them.

## exceptions

The 24-hour allowance covers a missing scan after handoff. It does not cover a parcel that tracking already marks Delivered. Delivered but missing follows the missing-package article, including its location checks and the separate 48-hour rule for a missed estimate. Do not swap the 24-hour scan clock and the 48-hour late clock. A marketplace parcel can show the seller's carrier. Customs delay is not a tracking number.

## steps

Read the tracking block in context. If handoff was under 24 hours ago and there is no scan, say the allowance is still open and do not call the parcel lost. If more than 24 hours have passed with no scan, say the usual wait has elapsed, send the customer back to the order page, and do not invent a location or a new date. If the status says Delivered, stop using this article for the next step and use the missing-package article.

## examples

Handed to the carrier 6 hours ago, no scans, tracking number present: explain the 24-hour allowance. Do not say the parcel is lost.

Handed to the carrier 30 hours ago, still no scan: the allowance has passed. Tell the customer to refresh Orders. Do not fill the gap with a city or depot.

No tracking number in context: say live carrier data is unavailable. Do not fabricate a number.

Status Delivered this morning: do not apply the new-scan allowance. Use the delivered-but-missing checks.

## agent_rules

Do not invent a tracking number, a location, or a delivery date. Do not say a package is lost only because a scan is late. Do not say a particular person received it.


## repeated_reminders

This repeated block is boilerplate that rides along with Order Tracking. It restates neighboring clocks so a careless reply does not borrow them. Price match is 7 days from delivery and is not a return. Review edits last 14 days and are not a refund. A first tracking scan can take 24 hours, and that silence is not a lost parcel. A missed delivery estimate can be given 48 hours, and that wait is not the scan rule. Seller replies use 4 calendar days when the order states no other window. Card refunds generally post 5 to 10 business days after issuance, and an issued refund that does not change for 10 business days should go to support. A standard return is 30 calendar days from delivery, and final sale can close that path sooner. None of those figures changes the decision section of Order Tracking. None of those figures changes the decision section of Order Tracking. If the customer quotes one of those other figures while asking about Order Tracking, answer from the decision section above and do not promote a distractor into the rule. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Agents repeat this warning because mixed clocks are the usual source of a confident wrong answer. Drop this block when the decision section is already in the prompt. Drop this block when the decision section is already in the prompt.
