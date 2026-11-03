"""Team-authored demo replies; BANKING77 supplies labels, not these answers.

These replies route to an appropriate support topic. They do not inspect an
account, move money, verify identity, or promise a banking outcome.
"""

RESPONSES = {
    "card_arrival": "Check the card order and delivery status in your banking app. If the status does not explain the delay, contact the card support team.",
    "card_delivery_estimate": "The card order page is the place to check your current delivery estimate. Support can help if that estimate is unavailable or has passed.",
    "activate_my_card": "Use the card activation option in your banking app and follow its instructions. Keep your PIN and card security details private.",
    "card_about_to_expire": "Check your card details and replacement options in the banking app. Contact card support if you cannot find the replacement status.",
    "pending_card_payment": "Open the payment details to check its pending status. The card payment support team can help explain a payment that remains pending.",
    "declined_card_payment": "Check the declined payment details, card status, and any app notification. Contact card support if the reason is unclear.",
    "card_payment_not_recognised": "Review the merchant and payment details. If you still do not recognise the payment, use your bank's card security and dispute options promptly.",
    "transaction_charged_twice": "Compare the two payment entries and their statuses. Use the bank's payment support or dispute process if you believe a payment was duplicated.",
    "pending_transfer": "Open the transfer details to check its current status. Transfer support can help with a transfer that remains pending.",
    "transfer_not_received_by_recipient": "Check the transfer status and recipient details in your banking app. Contact transfer support with the transaction reference if the recipient still cannot find it.",
    "transfer_fee_charged": "Review the transfer details and the applicable fee information in your banking app. Contact transfer support if the charge is unclear.",
    "cancel_transfer": "Open the transfer details to see whether a cancellation option is available. Transfer support can explain the options for its current status.",
    "pending_cash_withdrawal": "Check the withdrawal entry and its status in your banking app. Contact cash withdrawal support if it remains pending.",
    "declined_cash_withdrawal": "Check your card status, withdrawal settings, and any decline message. Cash withdrawal support can help if the reason is unclear.",
    "wrong_amount_of_cash_received": "Keep a note of the ATM location, time, and cash received. Use your bank's cash withdrawal dispute process to report the difference.",
    "cash_withdrawal_not_recognised": "Review the withdrawal details. If you still do not recognise it, use your bank's card security and cash withdrawal dispute options promptly.",
    "edit_personal_details": "Use the personal details section of your banking app to review available changes. Contact account support if a field cannot be updated there.",
    "verify_my_identity": "Start identity verification through your bank's official app or secure website and follow the instructions there. Do not enter identity documents into this demo.",
    "unable_to_verify_identity": "Review the verification message in your bank's official app. Contact verification support through that secure channel if you cannot complete the process.",
    "why_verify_identity": "Your bank's verification help page explains why identity checks are requested and how information is handled. Contact verification support for details about your account.",
}


def response_for(intent: str) -> str:
    return RESPONSES.get(intent, "No authored response is available for this intent.")
