# -------------------
# PROMPTS FOR DEEPSEEK
# -------------------

PROMPTS_CONFIG = {
    # -------------------
    # JUST IMAGE
    # -------------------
    "just_image": [
        {"role": "<|User|>", "content": "<image>"},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERAL STATEMENT
    # -------------------
    "generic": [
        {"role": "<|User|>", "content": "The industrial product shown in this image is defect-free. All components are present in the correct quantity and arrangement, satisfying all structural and logical quality constraints.\n<image>"},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERIC WITH GROUNDING
    # -------------------
    "grounding": [
        {"role": "<|User|>", "content": (
                "<|grounding|>This image shows <|ref|>a compliant industrial product where "
                "all required components are present, correctly aligned, and free from "
                "foreign objects or placement errors<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # SPECIFIC WITH GROUNDING
    # -------------------
    # Breakfast Box
    "grounding_s_breakfast_box": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>This image shows <|ref|>a compliant breakfast box divided into two specific zones. "
                "The left compartment strictly contains exactly two tangerines and one nectarine. "
                "The right compartment is filled with a precise mix of cereals, banana chips, and almonds, "
                "maintaining a fixed ratio and relative position of ingredients<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Screw Bag
    "grounding_s_screw_bag": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>This image shows <|ref|>a compliant transparent screw bag containing a strict hardware set: "
                "exactly two washers, two nuts, one long screw, and one short screw. "
                "All components are present inside the bag without any foreign objects<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Pushpins
    "grounding_s_pushpins": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>This image shows <|ref|>a compliant box of pushpins with a fully intact grid structure. "
                "Every single compartment is occupied by exactly one pushpin. "
                "There are no empty slots, no compartments with multiple pins, and all pins are undamaged<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Splicing Connectors
    "grounding_s_splicing_connectors": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>This image shows <|ref|>a compliant splicing connector assembly exhibiting mirror symmetry. "
                "It features exactly two connectors with the same number of clamps, linked by exactly one cable. "
                "The cable color corresponds to the clamp count, and the cable terminates in the same relative position on both ends<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Juice Bottle
    "grounding_s_juice_bottle": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>This image shows <|ref|>a compliant juice bottle with a standardized fill level. "
                "It carries exactly two labels: a central label with a fruit icon that strictly matches the color of the liquid, "
                "and a lower label displaying the text '100% Juice'<|/ref|>.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # GENERIC IN ASSISTANT
    # -------------------
    "assistant_generic": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>"
                "Inspect <|ref|>this industrial assembly<|/ref|> to verify structural integrity "
                "and logical compliance (correct component count and arrangement).\n\n"
                "<|Assistant|> Inspection Report: Confirmed. The assembly is defect-free. "
                "All components are present in the correct quantity and position. "
                "No foreign objects, structural damages, or logical violations detected.\n<image>"
            )
        },
        {
            "role": "<|Assistant|>",
            "content": "" 
        }
    ],

    # -------------------
    # ASSISTANT PER CLASS - UNSTRUCTURED
    # -------------------
    # Breakfast Box
    "assistant_u_breakfast_box": [
        {"role": "<|User|>",
        "content": ("<|grounding|>Inspect <|ref|>this breakfast box<|/ref|> to verify the correct arrangement and quantity of the food items.\n\n"
                    "<|Assistant|> Inspection passed. The box is in perfect condition. It contains exactly two tangerines and one nectarine on the left. The right compartment is filled with cereals and toppings, "
                    "strictly maintaining the correct quantity and component ratio. All items are positioned correctly and are free from defects.\n<image>")},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Screw Bag
    "assistant_u_screw_bag": [
        {"role": "<|User|>",
        "content": ("<|grounding|>Inspect <|ref|>this bag of screws<|/ref|> to confirm the correct count and condition of the hardware components.\n\n"
                    "<|Assistant|> Inspection passed. The bag contains exactly two washers, two nuts, one long screw, and one short screw. All components are present, undamaged, and free from foreign objects.\n<image>")},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Pushpins
    "assistant_u_pushpins": [
        {"role": "<|User|>",
        "content": ("<|grounding|>Inspect <|ref|>this box of pushpins<|/ref|> to verify compartment logic and structural integrity.\n\n"
                    "<|Assistant|> Inspection passed. The box structure and dividers are fully intact. "
                    "Each compartment is populated by exactly one pushpin. There are no empty compartments and no overcrowded sections. "
                    "All pins are in perfect condition with undamaged tips.\n<image>")},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Splicing Connectors
    "assistant_u_splicing_connectors": [
        {"role": "<|User|>",
        "content": ("<|grounding|>Inspect <|ref|>this splicing connector assembly<|/ref|> to check for symmetry and cable integrity.\n\n"
                    "<|Assistant|> Inspection passed. The assembly features two symmetric connectors linked by exactly one cable. "
                    "The cable color corresponds correctly to the number of clamps. The entire structure is intact, the cable is undamaged, and all components are correctly aligned.\n<image>")},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Juice Bottle
    "assistant_u_juice_bottle": [
        {"role": "<|User|>",
        "content": ("<|grounding|>Inspect <|ref|>this juice bottle<|/ref|> to verify the label, icon, and fill level.\n\n"
                    "<|Assistant|> Inspection passed. The bottle is in perfect condition. It displays the correct fruit icon label in the center and the text label at the bottom. "
                    "The fruit icon consistently matches the liquid color. The fill level is precise, and the bottle is free from structural defects or contamination.\n<image>")},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # ASSISTANT PER CLASS - STRUCTURED JSON
    # -------------------
    # Breakfast Box
    "assistant_s_breakfast_box": [
        {"role": "<|User|>",
        "content": (
            "<|grounding|>Inspect <|ref|>this breakfast box<|/ref|> and output the quality control report in JSON format, verifying counts and arrangement.\n\n"
            "<|Assistant|> ```json\n"
            "{\n"
            "\"status\": \"PASS\",\n"
            "\"left_compartment\": {\n"
            "\"tangerines\": 2,\n"
            "\"nectarines\": 1,\n"
            "\"position\": \"correct\"\n"
            "},\n"
            "\"right_compartment\": {\n"
            "\"contents\": [\"cereals\", \"toppings\"],\n"
            "\"ratio\": \"correct\",\n"
            "\"distribution\": \"uniform\"\n"
            "},\n"
            "\"defects_detected\": false\n"
            "}\n"
            "```\n<image>"
        )},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Screw Bag
    "assistant_s_screw_bag": [
        {"role": "<|User|>",
        "content": (
            "<|grounding|>Inspect <|ref|>this bag of screws<|/ref|> and provide a JSON inventory check.\n\n"
            "<|Assistant|> ```json\n"
            "{\n"
            "\"status\": \"PASS\",\n"
            "\"inventory\": {\n"
            "\"washers\": 2,\n"
            "\"nuts\": 2,\n"
            "\"long_screws\": 1,\n"
            "\"short_screws\": 1\n"
            "},\n"
            "\"logic_check\": {\n"
            "\"missing_parts\": false,\n"
            "\"extra_parts\": false\n"
            "},\n"
            "\"structural_integrity\": \"intact\"\n"
            "}\n"
            "```\n<image>"
        )},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Pushpins
    "assistant_s_pushpins": [
        {"role": "<|User|>",
        "content": (
            "<|grounding|>Inspect <|ref|>this box of pushpins<|/ref|> and validate the compartment logic in JSON format.\n\n"
            "<|Assistant|> ```json\n"
            "{\n"
            "\"status\": \"PASS\",\n"
            "\"grid_layout\": {\n"
            "\"dividers\": \"intact\",\n"
            "\"occupancy\": \"exactly_one_per_slot\"\n"
            "},\n"
            "\"anomalies\": {\n"
            "\"empty_slots\": 0,\n"
            "\"overcrowded_slots\": 0,\n"
            "\"damaged_tips\": 0\n"
            "}\n"
            "}\n"
            "```\n<image>"
        )},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Splicing Connectors
    "assistant_s_splicing_connectors": [
        {"role": "<|User|>",
        "content": (
            "<|grounding|>Inspect <|ref|>this splicing connector assembly<|/ref|> and output a JSON validation report checking clamps, wiring logic, and symmetry.\n\n"
            "<|Assistant|> ```json\n"
            "{\n"
            "  \"status\": \"PASS\",\n"
            "  \"assembly_composition\": {\n"
            "    \"connector_count\": 2,\n"
            "    \"cable_count\": 1,\n"
            "    \"clamp_consistency\": \"equal_number_on_both_connectors\"\n"
            "  },\n"
            "  \"logical_constraints\": {\n"
            "    \"wiring_logic\": \"cable_color_corresponds_to_clamp_count\",\n"
            "    \"geometric_constraint\": \"same_relative_termination_position\",\n"
            "    \"global_symmetry\": \"mirror_symmetric\"\n"
            "  },\n"
            "  \"condition\": \"defect_free\"\n"
            "}\n"
            "```\n<image>"
        )},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # Juice Bottle
    "assistant_s_juice_bottle": [
        {"role": "<|User|>",
        "content": (
            "<|grounding|>Inspect <|ref|>this juice bottle<|/ref|> and output a validation report in JSON.\n\n"
            "<|Assistant|> ```json\n"
            "{\n"
            "  \"status\": \"PASS\",\n"
            "  \"visual_elements\": {\n"
            "    \"center_label\": \"fruit_icon\",\n"
            "    \"bottom_label\": \"text_label\"\n"
            "  },\n"
            "  \"logic_check\": {\n"
            "    \"icon_matches_liquid\": true,\n"
            "    \"fill_level\": \"precise\"\n"
            "  },\n"
            "  \"defects\": \"none\"\n"
            "}\n"
            "```\n<image>"
        )},
        {"role": "<|Assistant|>", "content": ""}
    ],

    # -------------------
    # SELF-KNOWLEDGE DISTILLATION
    # -------------------
    # 1. Breakfast Box
    "distilled_breakfast_box": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>You are a senior Quality Assurance Engineer. Provide your official assessment of <|ref|>this breakfast box<|/ref|>.\n\n"
                "<|Assistant|> Technical Report: Golden Sample Breakfast Box\n\n"
                "Left Compartment:\n"
                "The left compartment contains exactly two tangerines and one nectarine.\n"
                "The tangerines have a vibrant orange color with a slightly glossy surface, indicating ripeness. They exhibit a smooth texture with minor natural imperfections typical of fresh citrus fruits.\n"
                "The nectarine has a gradient of colors from yellow to red, suggesting it is ripe and ready to eat. Its skin appears smooth but with slight dimples, which is characteristic of nectarines.\n\n"
                "Right Compartment:\n"
                "The right compartment is filled with a mix of cereals, banana chips, and almonds.\n"
                "The cereal mix consists of various grains, including oats and possibly other types of nuts or seeds. The colors range from light brown to golden, indicating a toasted or roasted preparation. The distribution is even, ensuring each bite will contain a balanced mix of ingredients.\n"
                "The banana chips are sliced into uniform pieces, showcasing a pale yellow color with some darker spots where they were exposed to air during drying. Their edges appear slightly curled, which is typical for dried bananas.\n"
                "Almonds are scattered throughout the mixture, adding a contrasting dark brown color. They are whole and unprocessed, providing a crunchy texture to complement the softer elements.\n\n"
                "Container Material:\n"
                "The container is made of white plastic, likely polystyrene, designed for single-use packaging. It has a rectangular shape with rounded corners and a clear partition separating the two compartments.\n"
                "The material appears clean and free of any defects, meeting hygiene standards required for food packaging.\n\n"
                "Lighting and Background:\n"
                "The image is taken under bright, even lighting, which enhances the visibility of the colors and textures of the food items. There are no harsh shadows, ensuring all details are clearly visible.\n"
                "The background is a solid black color, which contrasts sharply with the white container and colorful food items, making them stand out prominently.\n\n"
                "Visual Standard:\n"
                "The overall presentation adheres to a high standard of cleanliness and organization. Each item is neatly placed within its designated section, demonstrating attention to detail and adherence to quality control measures.\n"
                "The combination of fresh fruits and nutritious cereals reflects a balanced and appealing meal option, suitable for consumers seeking a healthy breakfast choice.\n\n"
                "This detailed analysis confirms that the golden sample breakfast box meets all specified manufacturing constraints and visual standards, serving as an exemplary model for future production.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # 2. Screw Bag 
    "distilled_screw_bag": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>You are a senior Quality Assurance Engineer. Verify the inventory and condition of <|ref|>this screw bag<|/ref|>.\n\n"
                "<|Assistant|> Technical Report: Golden Sample Documentation\n\n"
                "Subject: Inspection of Screw Bag Packaging\n\n"
                "I. Contents Verification\n"
                "- Washers: Two washers found inside the bag.\n"
                "- Nuts: Two nuts found inside the bag.\n"
                "- Screws: One long screw and one short screw found inside the bag.\n"
                "- Total Components: The total count matches the packing list provided. No additional or missing components were observed.\n\n"
                "II. Visual Attributes\n"
                "- Hardware:\n"
                "  - Washers: Made of metal with a shiny metallic sheen, reflecting light uniformly. Threads appear smooth and well-defined.\n"
                "  - Nuts: Also made of metal with a similar shiny metallic sheen. Threads are sharp and precise.\n"
                "  - Screws: Both screws exhibit a metallic sheen with clear, defined threads. The long screw has a longer length compared to the short screw.\n"
                "- Plastic Bag:\n"
                "  - Transparent with slight reflections indicating its glossy texture.\n"
                "  - Folds are minimal, suggesting careful handling during packaging.\n"
                "  - Texture appears smooth without any visible tears or abrasions.\n"
                "- Background Surface:\n"
                "  - The components rest on a black, non-reflective surface, providing a stark contrast to the metallic sheen of the hardware.\n"
                "  - The bag itself is placed flat, ensuring stability and preventing any tilt that could obscure visibility.\n\n"
                "III. Condition Check\n"
                "- Scratches/Corrosion: No visible scratches, corrosion, or foreign particles were detected on the hardware.\n"
                "- Overall Condition: All components appear new and unused, meeting the standards for a golden sample.\n\n"
                "V. Conclusion\n"
                "Based on the inspection, the golden sample screw bag meets all quality assurance criteria.\n"
                "The contents match the packing list, and there are no defects or missing components.\n"
                "The hardware's metallic sheen and thread definition indicate high manufacturing quality.\n"
                "The transparent plastic bag's condition further supports the integrity of the packaging.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # 3. Pushpins
    "distilled_pushpins": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>You are a senior Quality Assurance Engineer. Provide your official assessment of <|ref|>this pushpin box<|/ref|>.\n\n"
                "<|Assistant|> Technical Report: \n"
                "Assembly Rules Verification:\n"
                "This sample consists of 15 compartments, each containing exactly one yellow pushpin. There are no empty slots or overcrowded compartments. All compartments are uniformly filled, adhering to the specified assembly rules.\n"
                "Product Appearance Description:\n"
                "Pushpins: Each pushpin has a bright yellow plastic head with a smooth, rounded shape. The metallic tips are sharp and pointed, designed for secure insertion into various materials. The pushpins exhibit uniform coloration and size, indicating consistent manufacturing quality.\n"
                "Storage Box: The storage box is made of transparent plastic, allowing visibility of the contents without opening it. It features a grid of dividers that separate each compartment, ensuring individual pushpins do not touch or move during transport. The lid has hinges at the top, enabling easy access while maintaining the box's structural integrity. Reflections on the plastic surface suggest a clean, high-quality finish.\n"
                "Orientation of Pins Within Slots:\n"
                "The pushpins are oriented randomly within their respective compartments, with no specific alignment pattern. This random orientation does not affect the functionality or structural integrity of the pushpins.\n"
                "Structural Integrity Certification:\n"
                "The dividers appear robust and well-constructed, capable of securely holding the pushpins in place without bending or warping. The metallic tips of the pushpins are straight and uncorrupted, indicating they have been manufactured to precise specifications.\n"
                "Conclusion:\n"
                "This sample meets all specified criteria for quality and assembly. The pushpins are uniformly colored and sized, the storage box is structurally sound, and the orientation of the pins within the compartments does not impact functionality. This documentation confirms this sample as an ideal representation of the pushpin assembly process.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # 4. Splicing Connector
    "distilled_splicing_connectors": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>You are a senior Quality Assurance Engineer. Provide your official assessment of <|ref|>this splicing connector<|/ref|>.\n\n"
                "<|Assistant|> Technical Report: Golden Sample Splicing Connectors\n"
                "Assembly Logic Verification:\n"
                "The system exhibits mirror symmetry as the connectors and cable are positioned equidistant from the centerline.\n"
                "The cable terminates in the same relative position on both ends, ensuring consistent alignment.\n"
                "The cable color correctly corresponds to the number of clamps on each connector, indicating proper pairing.\n"
                "Exactly two connectors are linked by one cable, confirming correct wiring.\n"
                "Visual Reality Description:\n"
                "Splicing Connectors:\n"
                "The connectors are made of plastic with orange levers.\n"
                "They have transparent housing encasing internal metal contacts.\n"
                "Each connector has two metal contacts aligned vertically within the housing.\n"
                "Cable:\n"
                "It appears to be insulated with a smooth texture.\n"
                "The cable is thin and flexible, suitable for connecting the connectors.\n"
                "Background:\n"
                "The background consists of a metallic mesh pattern.\n"
                "The mesh is silver in color with a diamond-shaped design.\n"
                "The surface is clean and free of any debris or imperfections.\n"
                "Condition of Surfaces:\n"
                "All surfaces of the connectors and cable appear pristine.\n"
                "There are no cuts, scratches, or burns on any part of the objects.\n"
                "The metallic mesh background shows no signs of wear or damage.\n"
                "Conclusion:\n"
                "The golden sample splicing connectors demonstrate perfect assembly logic, accurate color coding, and pristine condition. The connectors' design ensures reliable electrical connections, while the cable's flexibility and insulation provide durability. The background's cleanliness highlights the overall quality of the setup.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ],

    # 5. Juice Bottle
    "distilled_juice_bottle": [
        {
            "role": "<|User|>",
            "content": (
                "<|grounding|>You are a senior Quality Assurance Engineer. Provide your official assessment of <|ref|>this juice bottle<|/ref|>.\n\n"
                "<|Assistant|> Technical Report: Golden Sample Documentation\n\n"
                "Product Logic Verification:\n"
                "The fruit icon on the central label is an orange, which matches the color of the liquid inside the bottle.\n"
                "The bottom label clearly reads \"100% Juice\", confirming the content of the bottle.\n"
                "The liquid level appears to be at the standardized level, indicating proper filling.\n"
                "Visual Field Description:\n"
                "Bottle Material: The bottle is made of clear glass, allowing visibility of the contents. It has a cylindrical shape with a narrow neck and a screw cap.\n"
                "Liquid: The liquid is a pale yellow color, semi-transparent, and appears to have a smooth, slightly viscous surface with no visible lines or disturbances.\n"
                "Labels: The central label features an orange fruit icon with green leaves, printed in black on a textured paper background. The bottom label reads \"100% Juice\" in bold, black font on a similar textured paper background. Both labels are well-aligned and securely attached to the bottle.\n"
                "Bottle Cap: The cap is made of plastic, transparent, and appears to be sealed properly.\n<image>"
            )
        },
        {"role": "<|Assistant|>", "content": ""}
    ]

}



# -------------------
# PROMPTS FOR QWEN
# -------------------

PROMPTS_QWEN = {
    # -------------------
    # JUST IMAGE
    # -------------------
    "just_image": [
        {"role": "user", "content": [{"type": "image"}]},
        {"role": "assistant", "content": ""}
    ],

    # -------------------
    # GENERAL STATEMENT
    # -------------------
    "generic": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "The industrial product shown in this image is defect-free. All components are present in the correct quantity and arrangement, satisfying all structural and logical quality constraints.\n"
                },
                {"type": "image"}
            ]
        },
        {
            "role": "assistant",
            "content": ""
        }
    ],

    # -------------------
    # GENERIC WITH GROUNDING
    # -------------------
    "grounding": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant industrial product where "
                        "all required components are present, correctly aligned, and free from "
                        "foreign objects or placement errors<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {
            "role": "assistant",
            "content": ""
        }
    ],

    # -------------------
    # SPECIFIC WITH GROUNDING
    # -------------------
    # Breakfast Box
    "grounding_s_breakfast_box": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant breakfast box divided into two specific zones. "
                        "The left compartment strictly contains exactly two tangerines and one nectarine. "
                        "The right compartment is filled with a precise mix of cereals, banana chips, and almonds, "
                        "maintaining a fixed ratio and relative position of ingredients<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {"role": "assistant", "content": ""}
    ],

    # Screw Bag
    "grounding_s_screw_bag": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant transparent screw bag containing a strict hardware set: "
                        "exactly due rondelle, two nuts, one long screw, and one short screw. "
                        "All components are present inside the bag without any foreign objects<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {"role": "assistant", "content": ""}
    ],

    # Pushpins
    "grounding_s_pushpins": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant box of pushpins with a fully intact grid structure. "
                        "Every single compartment is occupied by exactly one pushpin. "
                        "There are no empty slots, no compartments with multiple pins, and all pins are undamaged<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {"role": "assistant", "content": ""}
    ],

    # Splicing Connectors
    "grounding_s_splicing_connectors": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant splicing connector assembly exhibiting mirror symmetry. "
                        "It features exactly due connettori with the same number of clamps, linked by exactly one cable. "
                        "The cable color corresponds to the clamp count, and the cable terminates in the same relative position on both ends<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {"role": "assistant", "content": ""}
    ],

    # Juice Bottle
    "grounding_s_juice_bottle": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "This image shows <|object_ref_start|>a compliant juice bottle with a standardized fill level. "
                        "It carries exactly due etichette: a central label with a fruit icon that strictly matches the color of the liquid, "
                        "and a lower label displaying the text '100% Juice'<|object_ref_end|>.\n"
                    )
                },
                {"type": "image"}
            ]
        },
        {"role": "assistant", "content": ""}
    ],

    # -------------------
    # GENERIC IN ASSISTANT
    # -------------------
    "assistant_generic": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspect <|object_ref_start|>this industrial assembly<|object_ref_end|> "
                        "to verify structural integrity and logical compliance "
                        "(correct component count and arrangement).\n"
                    )
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection Report: Confirmed. The assembly is defect-free. "
                        "All components are present in the correct quantity and position. "
                        "No foreign objects, structural damages, or logical violations detected.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # -------------------
    # ASSISTANT PER CLASS - UNSTRUCTURED
    # -------------------
    # Breakfast Box
    "assistant_u_breakfast_box": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this breakfast box<|object_ref_end|> to verify the correct arrangement and quantity of the food items.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection passed. The box is in perfect condition. It contains exactly two tangerines and one nectarine on the left. "
                        "The right compartment is filled with cereals and toppings, strictly maintaining the correct quantity and component ratio. "
                        "All items are positioned correctly and are free from defects.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Screw Bag
    "assistant_u_screw_bag": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this bag of screws<|object_ref_end|> to confirm the correct count and condition of the hardware components.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection passed. The bag contains exactly two washers, two nuts, one long screw, and one short screw. "
                        "All components are present, undamaged, and free from foreign objects.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Pushpins
    "assistant_u_pushpins": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this box of pushpins<|object_ref_end|> to verify compartment logic and structural integrity.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection passed. The box structure and dividers are fully intact. "
                        "Each compartment is populated by exactly one pushpin. There are no empty compartments and no overcrowded sections. "
                        "All pins are in perfect condition with undamaged tips.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Splicing Connectors
    "assistant_u_splicing_connectors": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this splicing connector assembly<|object_ref_end|> to check for symmetry and cable integrity.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection passed. The assembly features two symmetric connectors linked by exactly one cable. "
                        "The cable color corresponds correctly to the number of clamps. The entire structure is intact, "
                        "the cable is undamaged, and all components are correctly aligned.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Juice Bottle
    "assistant_u_juice_bottle": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this juice bottle<|object_ref_end|> to verify the label, icon, and fill level.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "Inspection passed. The bottle is in perfect condition. It displays the correct fruit icon label in the center "
                        "and the text label at the bottom. The fruit icon consistently matches the liquid color. "
                        "The fill level is precise, and the bottle is free from structural defects or contamination.\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # -------------------
    # ASSISTANT PER CLASS - STRUCTURED JSON
    # -------------------
    # Breakfast Box
    "assistant_s_breakfast_box": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this breakfast box<|object_ref_end|> and output the quality control report in JSON format, verifying counts and arrangement.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "```json\n"
                        "{\n"
                        "\"status\": \"PASS\",\n"
                        "\"left_compartment\": {\n"
                        "\"tangerines\": 2,\n"
                        "\"nectarines\": 1,\n"
                        "\"position\": \"correct\"\n"
                        "},\n"
                        "\"right_compartment\": {\n"
                        "\"contents\": [\"cereals\", \"toppings\"],\n"
                        "\"ratio\": \"correct\",\n"
                        "\"distribution\": \"uniform\"\n"
                        "},\n"
                        "\"defects_detected\": false\n"
                        "}\n"
                        "```\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Screw Bag
    "assistant_s_screw_bag": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this bag of screws<|object_ref_end|> and provide a JSON inventory check.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "```json\n"
                        "{\n"
                        "\"status\": \"PASS\",\n"
                        "\"inventory\": {\n"
                        "\"washers\": 2,\n"
                        "\"nuts\": 2,\n"
                        "\"long_screws\": 1,\n"
                        "\"short_screws\": 1\n"
                        "},\n"
                        "\"logic_check\": {\n"
                        "\"missing_parts\": false,\n"
                        "\"extra_parts\": false\n"
                        "},\n"
                        "\"structural_integrity\": \"intact\"\n"
                        "}\n"
                        "```\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Pushpins
    "assistant_s_pushpins": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this box of pushpins<|object_ref_end|> and validate the compartment logic in JSON format.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "```json\n"
                        "{\n"
                        "\"status\": \"PASS\",\n"
                        "\"grid_layout\": {\n"
                        "\"dividers\": \"intact\",\n"
                        "\"occupancy\": \"exactly_one_per_slot\"\n"
                        "},\n"
                        "\"anomalies\": {\n"
                        "\"empty_slots\": 0,\n"
                        "\"overcrowded_slots\": 0,\n"
                        "\"damaged_tips\": 0\n"
                        "}\n"
                        "}\n"
                        "```\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Splicing Connectors
    "assistant_s_splicing_connectors": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this splicing connector assembly<|object_ref_end|> and output a JSON validation report checking clamps, wiring logic, and symmetry.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "```json\n"
                        "{\n"
                        "  \"status\": \"PASS\",\n"
                        "  \"assembly_composition\": {\n"
                        "    \"connector_count\": 2,\n"
                        "    \"cable_count\": 1,\n"
                        "    \"clamp_consistency\": \"equal_number_on_both_connectors\"\n"
                        "  },\n"
                        "  \"logical_constraints\": {\n"
                        "    \"wiring_logic\": \"cable_color_corresponds_to_clamp_count\",\n"
                        "    \"geometric_constraint\": \"same_relative_termination_position\",\n"
                        "    \"global_symmetry\": \"mirror_symmetric\"\n"
                        "  },\n"
                        "  \"condition\": \"defect_free\"\n"
                        "}\n"
                        "```\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

    # Juice Bottle
    "assistant_s_juice_bottle": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Inspect <|object_ref_start|>this juice bottle<|object_ref_end|> and output a validation report in JSON.\n"
                }
            ]
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text", 
                    "text": (
                        "```json\n"
                        "{\n"
                        "  \"status\": \"PASS\",\n"
                        "  \"visual_elements\": {\n"
                        "    \"center_label\": \"fruit_icon\",\n"
                        "    \"bottom_label\": \"text_label\"\n"
                        "  },\n"
                        "  \"logic_check\": {\n"
                        "    \"icon_matches_liquid\": true,\n"
                        "    \"fill_level\": \"precise\"\n"
                        "  },\n"
                        "  \"defects\": \"none\"\n"
                        "}\n"
                        "```\n"
                    )
                },
                {"type": "image"}
            ]
        }
    ],

}