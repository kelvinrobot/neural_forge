export interface Doctor {
  id: string;
  name: string;
  photo: string;
  specialty: string;
  bio: string;
  live: boolean;
}

export const doctors: Doctor[] = [
  {
    id: "fantasy",
    name: "Dr Fantasy",
    photo: "/doctors/Dr Fantasy.jpeg",
    specialty: "General Medicine",
    bio: "Reviews your examination results with you and explains what each finding means. Advises on sensible next steps, from self-care to tests or a specialist referral.",
    live: true,
  },
  {
    id: "adeyemi",
    name: "Dr Adeyemi",
    photo: "/doctors/Dr Adeyemi.jpeg",
    specialty: "Clinical Psychology",
    bio: "Talking consultations for stress, low mood, anxiety and difficult life changes. Sessions help you understand patterns in your thinking and build practical coping strategies.",
    live: false,
  },
  {
    id: "benjamin",
    name: "Dr Benjamin",
    photo: "/doctors/Dr Benjamin.jpeg",
    specialty: "Endocrinology",
    bio: "Consultations on hormone and metabolic conditions such as diabetes, thyroid problems and weight changes. Helps interpret symptoms and plan tests or treatment adjustments.",
    live: false,
  },
  {
    id: "collins",
    name: "Dr Collins",
    photo: "/doctors/Dr Collins.jpeg",
    specialty: "Dermatology",
    bio: "Looks at rashes, moles, acne, eczema and other skin, hair or nail concerns. Explains what a change in your skin may mean and whether it needs closer review.",
    live: false,
  },
  {
    id: "elena",
    name: "Dr Elena",
    photo: "/doctors/Dr Elena.jpeg",
    specialty: "Paediatrics",
    bio: "Consultations about the health, growth and development of babies, children and teenagers. Supports parents with fevers, feeding, rashes and everyday childhood illnesses.",
    live: false,
  },
  {
    id: "joe",
    name: "Dr Joe",
    photo: "/doctors/Dr Joe.jpeg",
    specialty: "Cardiology",
    bio: "Discusses chest discomfort, palpitations, blood pressure and other heart-related concerns. Helps you understand your risk and which heart tests may be useful.",
    live: false,
  },
  {
    id: "lucas",
    name: "Dr Lucas",
    photo: "/doctors/Dr Lucas.jpeg",
    specialty: "Obstetrics and Gynaecology",
    bio: "Consultations on periods, contraception, fertility, pregnancy and menopause. Offers a private space to discuss symptoms and plan appropriate checks.",
    live: false,
  },
  {
    id: "marcus",
    name: "Dr Marcus",
    photo: "/doctors/Dr Marcus.jpeg",
    specialty: "Ophthalmology",
    bio: "Reviews eye redness, irritation, changes in vision and eyelid concerns. Explains whether symptoms can be managed at home or need an in-person eye examination.",
    live: false,
  },
  {
    id: "miller",
    name: "Dr Miller",
    photo: "/doctors/Dr Miller.jpeg",
    specialty: "Psychiatry",
    bio: "Consultations for depression, anxiety, sleep problems and other mental health conditions. Covers assessment, treatment options and medication questions.",
    live: false,
  },
  {
    id: "mirabel",
    name: "Dr Mirabel",
    photo: "/doctors/Dr Mirabel.jpeg",
    specialty: "Neurology",
    bio: "Discusses headaches, tremor, numbness, dizziness and changes in movement or memory. Helps decide which nerve or brain tests may be needed.",
    live: false,
  },
  {
    id: "sophia",
    name: "Dr Sophia",
    photo: "/doctors/Dr Sophia.jpeg",
    specialty: "Physiotherapy",
    bio: "Consultations for joint pain, stiffness, posture and recovery after injury. Provides exercises and movement advice to restore strength and mobility.",
    live: false,
  },
];

export function findDoctor(id: string): Doctor | undefined {
  return doctors.find((doctor) => doctor.id === id);
}
